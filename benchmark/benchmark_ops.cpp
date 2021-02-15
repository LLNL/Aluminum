////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "benchmark_utils.hpp"
#include <cxxopts.hpp>


template <AlOperation Op, typename Backend, typename T,
          std::enable_if_t<IsTypeSupported<Backend, T>::value, bool> = true>
void run_benchmark(cxxopts::ParseResult& parsed_opts) {
  if (!IsOpSupported<Op, Backend>::value) {
    std::cerr << "Backend does not support operator " << AlOperationName<Op> << std::endl;
    std::abort();
  }
  AlOperation op = Op;
  // Set up options.
  OpOptions<Backend> op_options;
  if (parsed_opts.count("inplace")) {
    op_options.inplace = true;
  }
  if (parsed_opts.count("nonblocking")) {
    op_options.nonblocking = true;
  }
  if (parsed_opts.count("reduction-op")) {
    op_options.reduction_op = get_reduction_op(
      parsed_opts["reduction-op"].as<std::string>());
  }
  op_options.root = parsed_opts["root"].as<int>();
  // TODO: Support multiple algorithms -- needs support in OpProfile.
  if (OpSupportsAlgos<Op>::value) {
    auto algorithms = get_algorithms<Backend>(
      op, parsed_opts["algorithm"].as<std::string>());
    if (algorithms.size() != 1) {
      std::cerr << "Can only benchmark one algorithm at a time." << std::endl;
      std::abort();
    }
    op_options.algos = algorithms[0];
  }

  auto sizes = get_sizes_from_opts(parsed_opts);

  CommWrapper<Backend> comm_wrapper(MPI_COMM_WORLD);
  OpProfile<Op, Backend, T> profile(comm_wrapper.comm(), op_options);
  Timer<Backend> timer;

  bool participates_in_pt2pt = true;
  if (IsPt2PtOp<Op>::value) {
    if (comm_wrapper.size() == 1) {
      std::cerr << "Cannot benchmark point-to-point with a single rank" << std::endl;
      std::abort();
    }
    // If there is an odd number of ranks, the last one sits out.
    if (!((comm_wrapper.size() % 2 != 0) &&
          (comm_wrapper.rank() == comm_wrapper.size() - 1))) {
      // Even ranks send to rank + 1, odd ranks receive from rank - 1.
      // If this is not sendrecv, we need to adjust the op.
      if (comm_wrapper.comm().rank() % 2 == 0) {
        op_options.src = comm_wrapper.rank() + 1;
        op_options.dst = comm_wrapper.rank() + 1;
        if (op != AlOperation::sendrecv) {
          op = AlOperation::send;
        }
      } else {
        op_options.src = comm_wrapper.rank() - 1;
        op_options.dst = comm_wrapper.rank() - 1;
        if (op != AlOperation::sendrecv) {
          op = AlOperation::recv;
        }
      }
    } else {
      participates_in_pt2pt = false;
    }
  }

  size_t num_iters = parsed_opts["num-iters"].as<size_t>();
  size_t num_warmup = parsed_opts["num-warmup"].as<size_t>();

  for (const auto& size : sizes) {
    if (IsVectorOp<Op>::value) {
      op_options.send_counts = std::vector<size_t>(comm_wrapper.size(), size);
      op_options.send_displs = Al::excl_prefix_sum(op_options.send_counts);
      op_options.recv_counts = op_options.send_counts;
      op_options.recv_displs = op_options.send_displs;
    }

    OpDispatcher<Backend, T> op_runner(op, op_options);
    size_t in_size = op_runner.get_input_size(size, comm_wrapper.comm());
    size_t out_size = op_runner.get_output_size(size, comm_wrapper.comm());
    // Ensure sizes are reasonable for MPI.
    if (!Al::internal::mpi::check_count_fits_mpi(size)
        || !Al::internal::mpi::check_count_fits_mpi(out_size)) {
      std::cout << "Input size " << size << " or output size " << out_size
                << " too large for MPI, skipping this and future sizes"
                << std::endl;
      break;
    }

    typename VectorType<T, Backend>::type input =
      VectorType<T, Backend>::gen_data(in_size);
    typename VectorType<T, Backend>::type output =
      VectorType<T, Backend>::gen_data(out_size);

    if (!IsPt2PtOp<Op>::value || participates_in_pt2pt) {
      for (size_t trial = 0; trial < num_warmup + num_iters; ++trial) {
        MPI_Barrier(MPI_COMM_WORLD);
        timer.start_timer(comm_wrapper.comm());
        op_runner.run(input, output, comm_wrapper.comm());
        if (op_options.nonblocking) {
          Al::Wait<Backend>(op_options.req);
        }
        double t = timer.end_timer(comm_wrapper.comm());
        if (trial >= num_warmup) {
          profile.add_result(size, t);
        }
      }
    } else if (IsPt2PtOp<Op>::value && !participates_in_pt2pt) {
      // These ranks still need to participate in the barriers.
      for (size_t trial = 0; trial < num_warmup + num_iters; ++trial) {
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (parsed_opts.count("summarize")) {
    auto summaries = profile.get_summary_stats(
      parsed_opts["summarize"].as<int>());
    if (comm_wrapper.rank() == 0) {
      std::cout << "Size Mean Median Stdev Min Max" << std::endl;
      for (const auto& p : summaries) {
        std::cout << p.first << " " << p.second << std::endl;
      }
    }
  }
  if (parsed_opts.count("save-to-file")) {
    profile.save_results(parsed_opts["save-to-file"].as<std::string>());
  }
  if (!parsed_opts.count("no-print-table")) {
    profile.print_results();
  }
}

template <AlOperation Op, typename Backend, typename T,
          std::enable_if_t<!IsTypeSupported<Backend, T>::value, bool> = true>
void run_benchmark(cxxopts::ParseResult& parsed_opts) {
  std::cerr << "Backend "
            << parsed_opts["backend"].as<std::string>()
            << " does not support datatype "
            << parsed_opts["datatype"].as<std::string>() << std::endl;
  std::abort();
}

template <typename Backend, typename T>
struct benchmark_op_functor {
  cxxopts::ParseResult& parsed_opts;
  benchmark_op_functor(cxxopts::ParseResult& parsed_opts_) :
    parsed_opts(parsed_opts_) {}
  template <AlOperation Op>
  void operator()() {
    run_benchmark<Op, Backend, T>(parsed_opts);
  }
};

struct benchmark_dispatcher {
  template <typename Backend, typename T>
  void operator()(cxxopts::ParseResult& parsed_opts) {
    auto op_str = parsed_opts["op"].as<std::string>();
    if (!is_operator_name(op_str)) {
      std::cerr << "Unknown operator " << op_str << std::endl;
      std::abort();
    }
    AlOperation op = op_str_to_op(op_str);
    call_op_functor(op, benchmark_op_functor<Backend, T>(parsed_opts));
  }
};

int main(int argc, char** argv) {
  test_init_aluminum(argc, argv);

  cxxopts::Options options("benchmark_ops", "Benchmark Aluminum operations");
  options.add_options()
    ("op", "Operator to benchmark", cxxopts::value<std::string>())
    ("backend", "Aluminum backend", cxxopts::value<std::string>())
    ("inplace", "Use an inplace operator")
    ("nonblocking", "Use a non-blocking operator")
    ("reduction-op", "Reduction operator to use (if needed)", cxxopts::value<std::string>())
    ("algorithm", "Operator algorithm to use", cxxopts::value<std::string>()->default_value(""))
    ("root", "Root of operator (if needed)", cxxopts::value<int>()->default_value("0"))
    ("size", "Size of message to test (roughly, the size sent to each process)", cxxopts::value<size_t>())
    ("min-size", "Minimum size of message to test", cxxopts::value<size_t>()->default_value("1"))
    ("max-size", "Maximum size of message to test", cxxopts::value<size_t>()->default_value("4194304"))
    ("datatype", "Message datatype", cxxopts::value<std::string>()->default_value("float"))
    ("num-iters", "Number of benchmark iterations", cxxopts::value<size_t>()->default_value("100"))
    ("num-warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value("10"))
    ("save-to-file", "Save results to a file", cxxopts::value<std::string>())
    ("summarize", "Print stats summary over all ranks or a specific rank", cxxopts::value<int>()->default_value("-1"))
    ("no-print-table", "Do not print results table")
    ("help", "Print help");
  auto parsed_opts = options.parse(argc, argv);

  if (parsed_opts.count("help")) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::cout << options.help() << std::endl;
    }
    test_fini_aluminum();
    std::exit(0);
  }

  // Simple validation.
  if (!parsed_opts.count("op")) {
    std::cerr << "Must provide an operator to test" << std::endl;
    test_fini_aluminum();
    return EXIT_FAILURE;
  }
  if (!parsed_opts.count("backend")) {
    std::cerr << "Must provide a backend to use" << std::endl;
    test_fini_aluminum();
    return EXIT_FAILURE;
  }

  dispatch_to_backend(parsed_opts, benchmark_dispatcher());

  test_fini_aluminum();
  return 0;
}
