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

#include "Al.hpp"
#include <iostream>
#include <sstream>
#include <cxxopts.hpp>
#include "test_utils.hpp"


template <typename T>
void print_vector(std::ostream& os, std::vector<T> v,
                  size_t start = 0, size_t end = std::numeric_limits<size_t>::max()) {
  std::stringstream ss;
  ss << "{";
  bool first = true;
  if (end > v.size()) {
    end = v.size();
  }
  for (size_t i = start; i < end; ++i) {
    if (first) {
      ss << v[i];
      first = false;
    } else {
      ss << ", " << v[i];
    }
  }
  ss << "}";
  os << ss.str();
}

template <typename Backend, typename T,
          std::enable_if_t<IsTypeSupportedByMPI<T>::value, bool> = true>
void do_mpi_gather(const T* input, T* output, size_t count,
                   typename Backend::comm_type& comm) {
  MPI_Gather(input, count, Al::internal::mpi::TypeMap<T>(),
             output, count, Al::internal::mpi::TypeMap<T>(),
             0, comm.get_comm());
}
template <typename Backend, typename T,
          std::enable_if_t<!IsTypeSupportedByMPI<T>::value, bool> = true>
void do_mpi_gather(const T* input, T* output, size_t count,
                   typename Backend::comm_type& comm) {
  const size_t num_bytes = count * sizeof(T);
  MPI_Gather(input, num_bytes, MPI_BYTE,
             output, num_bytes, MPI_BYTE,
             0, comm.get_comm());
}

template <typename Backend, typename T>
void print_vectors_from_root(std::vector<T>& v,
                             typename Backend::comm_type& comm,
                             std::string prefix = "") {
  if (comm.rank() == 0) {
    std::cout << prefix;
    std::vector<T> gathered_v(v.size() * comm.size());
    do_mpi_gather<Backend>(v.data(), gathered_v.data(), v.size(), comm);
    for (int rank = 0; rank < comm.size(); ++rank) {
      std::cout << rank << ": ";
      print_vector(std::cout, gathered_v, rank*v.size(), (rank+1)*v.size());
      std::cout << std::endl;
    }
  } else {
    do_mpi_gather<Backend, T>(v.data(), nullptr, v.size(), comm);
  }
  MPI_Barrier(comm.get_comm());
}

template <typename Backend, typename T>
void dump_data(typename VectorType<T, Backend>::type& input,
               typename VectorType<T, Backend>::type& output,
               std::vector<T>& mpi_input,
               std::vector<T>& mpi_output,
               typename Backend::comm_type& comm) {
  std::vector<T> host_input = VectorType<T, Backend>::copy_to_host(input);
  std::vector<T> host_output = VectorType<T, Backend>::copy_to_host(output);
  print_vectors_from_root<Backend>(host_input, comm, "Input:\n");
  print_vectors_from_root<Backend>(mpi_input, comm, "MPI input:\n");
  print_vectors_from_root<Backend>(host_output, comm, "Output:\n");
  print_vectors_from_root<Backend>(mpi_output, comm, "MPI output:\n");
}

template <typename Backend, typename T,
          std::enable_if_t<IsTypeSupported<Backend, T>::value, bool> = true>
void run_test(cxxopts::ParseResult& parsed_opts) {
  auto op_str = parsed_opts["op"].as<std::string>();
  if (!is_operator_name(op_str)) {
    std::cerr << "Unknown operator " << op_str << std::endl;
    std::abort();
  }
  AlOperation op = op_str_to_op(op_str);
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

  size_t min_size = parsed_opts["min-size"].as<size_t>();
  size_t max_size = parsed_opts["max-size"].as<size_t>();
  if (parsed_opts.count("size")) {
    min_size = parsed_opts["size"].as<size_t>();
    max_size = min_size;
  }
  auto sizes = get_sizes(min_size, max_size);

  CommWrapper<Backend> comm_wrapper(MPI_COMM_WORLD);

  bool participates_in_pt2pt = true;
  if (is_pt2pt_op(op)) {
    if (comm_wrapper.comm.size() == 1) {
      std::cerr << "Cannot test point-to-point with a single rank" << std::endl;
      std::abort();
    }
    // If there is an odd number of ranks, the last one sits out.
    if (!((comm_wrapper.comm.size() % 2 != 0) &&
          (comm_wrapper.comm.rank() == comm_wrapper.comm.size() - 1))) {
      // Even ranks send to rank + 1, odd ranks receive from rank - 1.
      // If this is not sendrecv, we need to adjust the op.
      if (comm_wrapper.comm.rank() % 2 == 0) {
        op_options.src = comm_wrapper.comm.rank() + 1;
        op_options.dst = comm_wrapper.comm.rank() + 1;
        if (op != AlOperation::sendrecv) {
          op = AlOperation::send;
        }
      } else {
        op_options.src = comm_wrapper.comm.rank() - 1;
        op_options.dst = comm_wrapper.comm.rank() - 1;
        if (op != AlOperation::sendrecv) {
          op = AlOperation::recv;
        }
      }
    } else {
      participates_in_pt2pt = false;
    }
  }
  for (const auto &size : sizes) {
    // Set up counts and displacements for vector operations.
    // TODO: Generalize to support more complex counts/displacements.
    if (is_vector_op(op)) {
      op_options.send_counts = std::vector<size_t>(comm_wrapper.comm.size(), size);
      op_options.send_displs = Al::excl_prefix_sum(op_options.send_counts);
      op_options.recv_counts = op_options.send_counts;
      op_options.recv_displs = op_options.send_displs;
    }
    OpDispatcher<Backend, T> op_runner(op, op_options);
    // The size is the amount each processor sends to another processor.
    // (Roughly equivalent to the sendcount parameter in MPI.)
    size_t in_size = op_runner.get_input_size(size, comm_wrapper.comm);
    // Get output buffer size.
    size_t out_size = op_runner.get_output_size(size, comm_wrapper.comm);
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
    std::vector<T> mpi_input = VectorType<T, Backend>::copy_to_host(input);
    std::vector<T> mpi_output = VectorType<T, Backend>::copy_to_host(output);
    // Save originals when in-place if we might print an error.
    typename VectorType<T, Backend>::type orig_input;
    std::vector<T> orig_mpi_input;
    if (op_options.inplace && parsed_opts.count("dump-on-error")) {
      orig_input = output;
      orig_mpi_input = mpi_output;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_pt2pt_op(op) || participates_in_pt2pt) {
      op_runner.run(input, output, comm_wrapper.comm);
      if (op_options.nonblocking) {
        Al::Wait<Backend>(op_options.req);
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (!is_pt2pt_op(op) || participates_in_pt2pt) {
      op_runner.run_mpi(mpi_input, mpi_output, comm_wrapper.comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    bool err = false;
    if (!op_options.inplace && !check_vector(mpi_input, input)) {
      std::cerr << comm_wrapper.comm.rank() << ": input does not match for size "
                << size << std::endl;
      err = true;
    }
    if (!check_vector(mpi_output, output)) {
      std::cerr << comm_wrapper.comm.rank() << ": output does not match for size "
                << size << std::endl;
      err = true;
    }
    // Check if any process reported an error.
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_BYTE, MPI_LOR,
                  comm_wrapper.comm.get_comm());
    if (err) {
      if (parsed_opts.count("dump-on-error")) {
        if (op_options.inplace) {
          dump_data<Backend>(orig_input, output, orig_mpi_input, mpi_output,
                             comm_wrapper.comm);
        } else {
          dump_data<Backend>(input, output, mpi_input, mpi_output,
                             comm_wrapper.comm);
        }
      }
      std::abort();
    }
  }
}

template <typename Backend, typename T,
          std::enable_if_t<!IsTypeSupported<Backend, T>::value, bool> = true>
void run_test(cxxopts::ParseResult& parsed_opts) {
  std::cerr << "Backend "
            << parsed_opts["backend"].as<std::string>()
            << " does not support datatype "
            << parsed_opts["datatype"].as<std::string>() << std::endl;
  std::abort();
}

template <typename Backend>
void dispatch_test_to_backend_and_type(cxxopts::ParseResult& parsed_opts) {
  const std::unordered_map<std::string, std::function<void()>> dispatch = {
    {"char", [&]() { run_test<Backend, char>(parsed_opts); } },
    {"schar", [&]() { run_test<Backend, signed char>(parsed_opts); } },
    {"uchar", [&]() { run_test<Backend, unsigned char>(parsed_opts); } },
    {"short", [&]() { run_test<Backend, short>(parsed_opts); } },
    {"ushort", [&]() { run_test<Backend, unsigned short>(parsed_opts); } },
    {"int", [&]() { run_test<Backend, int>(parsed_opts); } },
    {"uint", [&]() { run_test<Backend, unsigned int>(parsed_opts); } },
    {"long", [&]() { run_test<Backend, long>(parsed_opts); } },
    {"ulong", [&]() { run_test<Backend, unsigned long>(parsed_opts); } },
    {"longlong", [&]() { run_test<Backend, long long>(parsed_opts); } },
    {"ulonglong", [&]() { run_test<Backend, unsigned long long>(parsed_opts); } },
    {"float", [&]() { run_test<Backend, float>(parsed_opts); } },
    {"double", [&]() { run_test<Backend, double>(parsed_opts); } },
    {"longdouble", [&]() { run_test<Backend, long double>(parsed_opts); } },
#ifdef AL_HAS_NCCL
    {"half", [&]() { run_test<Backend, __half>(parsed_opts); } },
#endif
  };
  auto datatype = parsed_opts["datatype"].as<std::string>();
  auto i = dispatch.find(datatype);
  if (i == dispatch.end()) {
    std::cerr << "Unknown datatype " << datatype << std::endl;
    std::abort();
  }
  i->second();
}

void dispatch_test_to_backend(cxxopts::ParseResult& parsed_opts) {
  const std::unordered_map <std::string, std::function<void()>> dispatch = {
    {"mpi", [&](){ dispatch_test_to_backend_and_type<Al::MPIBackend>(parsed_opts); } },
#ifdef AL_HAS_NCCL
    {"nccl", [&](){ dispatch_test_to_backend_and_type<Al::NCCLBackend>(parsed_opts); } },
#endif
#ifdef AL_HAS_HOST_TRANSFER
    {"ht", [&](){ dispatch_test_to_backend_and_type<Al::HostTransferBackend>(parsed_opts); } },
#endif
  };
  auto backend = parsed_opts["backend"].as<std::string>();
  auto i = dispatch.find(backend);
  if (i == dispatch.end()) {
    std::cerr << "Unsupported backend " << backend << std::endl;
    std::abort();
  }
  i->second();
}


int main(int argc, char** argv) {
  test_init_aluminum(argc, argv);

  cxxopts::Options options("test_ops", "Compare Aluminum operators with MPI");
  options.add_options()
    ("op", "Operator to test", cxxopts::value<std::string>())
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
    ("dump-on-error", "Dump vectors on error")
    ("hang-rank", "Hang a specific or all ranks at startup", cxxopts::value<int>()->default_value("-1"));
  auto parsed_opts = options.parse(argc, argv);

  if (parsed_opts.count("hang-rank")) {
    int hang_rank = parsed_opts["hang-rank"].as<int>();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (hang_rank < 0 || rank == hang_rank) {
      if (hang_rank < 0 && rank == 0) {
        std::cout << "Hanging all ranks" << std::endl;
      } else if (rank == hang_rank) {
        std::cout << "Hanging rank " << rank << std::endl;
      }
      volatile bool hang = true;
      while (hang) {}
    }
  }

  // Simple validation.
  if (!parsed_opts.count("op")) {
    std::cerr << "Must provide an operator to test" << std::endl;
    std::abort();
  }
  if (!parsed_opts.count("backend")) {
    std::cerr << "Must provide a backend to use" << std::endl;
    std::abort();
  }

  dispatch_test_to_backend(parsed_opts);

  test_fini_aluminum();
}
