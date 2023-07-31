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
#include <iomanip>
#include <cxxopts.hpp>
#include "op_dispatcher.hpp"


struct BandwidthResults {
  // Store in separate arrays to simplify communication.
  /** Ranks initiating sends. */
  std::vector<int> srcs;
  /** Ranks receiving data. */
  std::vector<int> dsts;
  /** Sizes of send/receive buffers. */
  std::vector<size_t> sizes;
  /** Runtimes for each trial. */
  std::vector<double> times;
};


/** Manage benchmarking results. */
template <typename Backend, typename T>
class BandwidthResultsManager {
public:

  BandwidthResultsManager(typename Backend::comm_type& comm_,
                          size_t num_trials_) :
    comm(comm_), num_trials(num_trials_) {}

  void add_result(int src, int dst, size_t size, double time) {
    local_results.srcs.push_back(src);
    local_results.dsts.push_back(dst);
    local_results.sizes.push_back(size);
    local_results.times.push_back(time);
  }

  void print_results() {
    write_results(std::cout);
  }

  void save_results(std::string filename) {
    std::ofstream f(filename);
    if (f.fail()) {
      std::cerr << "Error opening " << filename << std::endl;
      std::abort();
    }
    write_results(f);
  }

  void write_results(std::ostream& os) {
    auto gathered_results = gather_results_to_root();
    if (comm.rank() != 0) {
      return;
    }
    // Print header.
    os << "Backend Type Src Dst Size Time\n";
    const std::string common_start =
      std::string(Al::AlBackendName<Backend>) + " "
      + std::string(typeid(T).name()) + " ";
    for (size_t i = 0; i < gathered_results.srcs.size(); ++i) {
      os << common_start
         << gathered_results.srcs[i] << " "
         << gathered_results.dsts[i] << " "
         << gathered_results.sizes[i] << " "
         << gathered_results.times[i] << "\n";
    }
  }

  void print_bandwidth_matrix(size_t size) {
    auto gathered_results = gather_results_to_root();
    if (comm.rank() != 0) {
      return;
    }
    // Build matrix of results. Indexing is [src][dst].
    // We accumulate the sum of the benchmarked times for a size, then
    // compute the mean of them. This, plus sizeof(T) and size is used
    // to compute the bandwidth.
    std::vector<std::vector<double>> times(comm.size());
    for (int i = 0; i < comm.size(); ++i) {
      times[i].resize(comm.size(), 0.0);
    }
    for (size_t i = 0; i < gathered_results.srcs.size(); ++i) {
      if (gathered_results.sizes[i] == size) {
        times[gathered_results.srcs[i]][gathered_results.dsts[i]] += gathered_results.times[i];
      }
    }

    std::cout << "src rank x destination rank; GiB/s for size "
              << size << " and type " << typeid(T).name() << "\n";
    // Print the matrix, converting times to bandwidths in GiB/s.
    const size_t size_bytes = size * sizeof(T);
    const size_t gb_denom = 1024 * 1024 * 1024;
    const int width = 7;
    // First row is header containing ranks.
    std::cout << std::setw(width) << " ";
    for (int rank = 0; rank < comm.size(); ++rank) {
      std::cout << std::setw(width) << rank << " ";
    }
    std::cout << "\n";
    for (int src = 0; src < comm.size(); ++src) {
      std::cout << std::setw(width) << src << " ";
      for (int dst = 0; dst < comm.size(); ++dst) {
        double mean_time = times[src][dst] / num_trials;
        double bw = (mean_time == 0) ? 0.0 : size_bytes / mean_time / gb_denom;
        std::cout << std::setprecision(width - 1)
                  << std::setw(width)
                  << bw << " ";
      }
      std::cout << "\n";
    }
  }

  // Print summary matrix of results.
private:
  typename Backend::comm_type& comm;
  BandwidthResults local_results;
  size_t num_trials;

  BandwidthResults gather_results_to_root() {
    if (comm.rank() == 0) {
      // Assumes every rank has run the same number of tests.
      BandwidthResults gathered_results;
      size_t num_results = local_results.srcs.size() * comm.size();
      gathered_results.srcs.resize(num_results);
      gathered_results.dsts.resize(num_results);
      gathered_results.sizes.resize(num_results);
      gathered_results.times.resize(num_results);
      MPI_Gather(local_results.srcs.data(), local_results.srcs.size(), MPI_INT,
                 gathered_results.srcs.data(), local_results.srcs.size(), MPI_INT,
                 0, comm.get_comm());
      MPI_Gather(local_results.dsts.data(), local_results.dsts.size(), MPI_INT,
                 gathered_results.dsts.data(), local_results.dsts.size(), MPI_INT,
                 0, comm.get_comm());
      MPI_Gather(local_results.sizes.data(), local_results.sizes.size(), Al::internal::mpi::TypeMap<size_t>(),
                 gathered_results.sizes.data(), local_results.sizes.size(), Al::internal::mpi::TypeMap<size_t>(),
                 0, comm.get_comm());
      MPI_Gather(local_results.times.data(), local_results.times.size(), MPI_DOUBLE,
                 gathered_results.times.data(), local_results.times.size(), MPI_DOUBLE,
                 0, comm.get_comm());
      return gathered_results;
    } else {
      MPI_Gather(local_results.srcs.data(), local_results.srcs.size(), MPI_INT,
                 nullptr, 0, MPI_INT,
                 0, comm.get_comm());
      MPI_Gather(local_results.dsts.data(), local_results.dsts.size(), MPI_INT,
                 nullptr, 0, MPI_INT,
                 0, comm.get_comm());
      MPI_Gather(local_results.sizes.data(), local_results.sizes.size(), Al::internal::mpi::TypeMap<size_t>(),
                 nullptr, 0, Al::internal::mpi::TypeMap<size_t>(),
                 0, comm.get_comm());
      MPI_Gather(local_results.times.data(), local_results.times.size(), MPI_DOUBLE,
                 nullptr, 0, MPI_DOUBLE,
                 0, comm.get_comm());
      return {};
    }
  }
};


/**
 * Run bandwidth test between src and dst.
 *
 * src sends data, dst receives data. Bandwidth recorded only on dst.
 */
template <typename Backend, typename T,
          std::enable_if_t<Al::IsTypeSupported<Backend, T>::value, bool> = true>
void benchmark_pair(int src, int dst,
                    CommWrapper<Backend>& comm_wrapper,
                    const std::vector<size_t>& sizes,
                    size_t num_iters, size_t num_warmup,
                    BandwidthResultsManager<Backend, T>& results) {
  Timer<Backend> timer;
  OpOptions<Backend> op_options;
  op_options.src = src;
  op_options.dst = dst;
  for (const auto& size : sizes) {
    typename VectorType<T, Backend>::type input =
      VectorType<T, Backend>::gen_data(size);
    typename VectorType<T, Backend>::type output =
      VectorType<T, Backend>::gen_data(size);

    for (size_t trial = 0; trial < num_warmup + num_iters; ++trial) {
      OpDispatcher<Backend, T> op_runner(
        (comm_wrapper.rank() == src) ? Al::AlOperation::send : Al::AlOperation::recv,
        op_options);
      timer.start_timer(comm_wrapper.comm());
      op_runner.run(input, output, comm_wrapper.comm());
      double t = timer.end_timer(comm_wrapper.comm());
      // Only receive side records time. (Send is local completion.)
      if (trial >= num_warmup && comm_wrapper.rank() == dst) {
        results.add_result(src, dst, size, t);
      }
    }
  }
}

template <typename Backend, typename T,
          std::enable_if_t<Al::IsTypeSupported<Backend, T>::value, bool> = true>
void run_benchmark(cxxopts::ParseResult& parsed_opts) {
  if (!Al::IsOpSupported<Al::AlOperation::send, Backend>::value
      || !Al::IsOpSupported<Al::AlOperation::recv, Backend>::value) {
    std::cerr << "Backend does not support send or recv" << std::endl;
    std::abort();
  }

  // Set up options.
  OpOptions<Backend> send_op_options;
  OpOptions<Backend> recv_op_options = send_op_options;

  auto sizes = get_sizes_from_opts(parsed_opts);
  size_t num_iters = parsed_opts["num-iters"].as<size_t>();
  size_t num_warmup = parsed_opts["num-warmup"].as<size_t>();

  StreamManager<Backend>::init(1UL);
  CommWrapper<Backend> comm_wrapper(MPI_COMM_WORLD);
  BandwidthResultsManager<Backend, T> results(comm_wrapper.comm(), num_iters);

  // Each rank will send to and receive from every rank.
  // To prevent interference, only one pair communicates at a time.
  for (int rank0 = 0; rank0 < comm_wrapper.size(); ++rank0) {
    for (int rank1 = 0; rank1 < comm_wrapper.size(); ++rank1) {
      if (rank0 == rank1) {
        continue;  // Not measuring local copy speed.
      }
      if (comm_wrapper.rank() == rank0 || comm_wrapper.rank() == rank1) {
        // Do bandwidth test between this pair.
        benchmark_pair<Backend, T>(
          rank0, rank1, comm_wrapper, sizes, num_iters, num_warmup, results);
      }
      // All ranks wait for this to complete.
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  if (parsed_opts.count("summarize")) {
    // Print using largest size, probably get the best bandwidth.
    results.print_bandwidth_matrix(sizes.back());
  }
  if (parsed_opts.count("save-to-file")) {
    results.save_results(parsed_opts["save-to-file"].as<std::string>());
  }
  if (parsed_opts.count("print-table")) {
    results.print_results();
  }

  StreamManager<Backend>::finalize();
}

template <typename Backend, typename T,
          std::enable_if_t<!Al::IsTypeSupported<Backend, T>::value, bool> = true>
void run_benchmark(cxxopts::ParseResult& parsed_opts) {
  std::cerr << "Backend "
            << parsed_opts["backend"].as<std::string>()
            << " does not support datatype "
            << parsed_opts["datatype"].as<std::string>() << std::endl;
  std::abort();
}

struct benchmark_dispatcher {
  template <typename Backend, typename T>
  void operator()(cxxopts::ParseResult& parsed_opts) {
    run_benchmark<Backend, T>(parsed_opts);
  }
};

int main(int argc, char** argv) {
  test_init_aluminum(argc, argv);

  cxxopts::Options options(
    "bandwidth", "Benchmark Aluminum latency and bandwidth");
  options.add_options()
    ("backend", "Aluminum backend", cxxopts::value<std::string>())
    ("size", "Size of message to use", cxxopts::value<size_t>())
    ("min-size", "Minimum size of message to test", cxxopts::value<size_t>()->default_value("1"))
    ("max-size", "Maximum size of message to test", cxxopts::value<size_t>()->default_value("4194304"))
    ("datatype", "Message datatype", cxxopts::value<std::string>()->default_value("float"))
    ("num-iters", "Number of benchmark iterations", cxxopts::value<size_t>()->default_value("100"))
    ("num-warmup", "Number of warmup iterations", cxxopts::value<size_t>()->default_value("10"))
    ("save-to-file", "Save results to a file", cxxopts::value<std::string>())
    ("summarize", "Print bandwidth matrix")
    ("print-table", "Print results table")
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

  if (!parsed_opts.count("backend")) {
    std::cerr << "Must provide a backend to use" << std::endl;
    test_fini_aluminum();
    return EXIT_FAILURE;
  }

  dispatch_to_backend(parsed_opts, benchmark_dispatcher());

  test_fini_aluminum();
  return 0;
}
