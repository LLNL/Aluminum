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

#include <iostream>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_ht_cuda.hpp"
#endif

size_t start_size = 1;
size_t max_size = 1<<30;
const size_t num_trials = 10;

template <typename Backend>
typename Backend::comm_type get_comm();

template <>
typename Al::MPIBackend::comm_type get_comm<Al::MPIBackend>() {
  return typename Al::MPIBackend::comm_type(MPI_COMM_WORLD);
}

#ifdef AL_HAS_CUDA
// Stream for communicators.
cudaStream_t bm_stream;
#endif

#ifdef AL_HAS_NCCL
template <>
typename Al::NCCLBackend::comm_type get_comm<Al::NCCLBackend>() {
  return typename Al::NCCLBackend::comm_type(MPI_COMM_WORLD, bm_stream);
}
#endif

#ifdef AL_HAS_MPI_CUDA
template <>
typename Al::MPICUDABackend::comm_type get_comm<Al::MPICUDABackend>() {
  return typename Al::MPICUDABackend::comm_type(MPI_COMM_WORLD, bm_stream);
}
#endif

#ifdef AL_HAS_HOST_TRANSFER
template <>
typename Al::HTBackend::comm_type get_comm<Al::HTBackend>() {
  return typename Al::HTBackend::comm_type(MPI_COMM_WORLD, bm_stream);
}
#endif

template <typename Backend>
void time_allreduce_algo(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::allreduce_algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    times.push_back(finish_timer<Backend>(comm));
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(in_place_input.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    in_place_times.push_back(finish_timer<Backend>(comm));
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " regular blocking ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " inplace blocking ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void time_nballreduce_algo(typename VectorType<Backend>::type input,
                           typename Backend::comm_type& comm,
                           typename Backend::allreduce_algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);    
    typename Backend::req_type req = get_request<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::NonblockingAllreduce<Backend>(
        input.data(), recv.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    Al::Wait<Backend>(req);
    times.push_back(finish_timer<Backend>(comm));
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::NonblockingAllreduce<Backend>(
        in_place_input.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    Al::Wait<Backend>(req);
    in_place_times.push_back(finish_timer<Backend>(comm));
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " regular nonblocking ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " inplace nonblocking ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void time_allreduce_algo_start(typename VectorType<Backend>::type input,
                               typename Backend::comm_type& comm,
                               typename Backend::allreduce_algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);    
    typename Backend::req_type req = get_request<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::NonblockingAllreduce<Backend>(
        input.data(), recv.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    times.push_back(finish_timer<Backend>(comm));
    Al::Wait<Backend>(req);
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::NonblockingAllreduce<Backend>(
        in_place_input.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    in_place_times.push_back(finish_timer<Backend>(comm));
    Al::Wait<Backend>(req);
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " regular start ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " inplace start ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void time_allreduce_algo_wait(typename VectorType<Backend>::type input,
                              typename Backend::comm_type& comm,
                              typename Backend::allreduce_algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);    
    typename Backend::req_type req = get_request<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    Al::NonblockingAllreduce<Backend>(
        input.data(), recv.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    start_timer<Backend>(comm);
    Al::Wait<Backend>(req);
    times.push_back(finish_timer<Backend>(comm));
    MPI_Barrier(MPI_COMM_WORLD);
    Al::NonblockingAllreduce<Backend>(
        in_place_input.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    start_timer<Backend>(comm);
    Al::Wait<Backend>(req);
    in_place_times.push_back(finish_timer<Backend>(comm));
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " regular wait ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " inplace wait ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void do_benchmark() {
  std::vector<typename Backend::allreduce_algo_type> algos
      = get_nb_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm = get_comm<Backend>();
  std::vector<size_t> sizes = {0};
  for (size_t size = start_size; size <= max_size; size *= 2) {
    sizes.push_back(size);
  }
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    // Benchmark algorithms.
    for (auto&& algo : algos) {
      time_allreduce_algo<Backend>(data, comm, algo);
      time_nballreduce_algo<Backend>(data, comm, algo);
      time_allreduce_algo_start<Backend>(data, comm, algo);
      time_allreduce_algo_wait<Backend>(data, comm, algo);
    }
  }
}


int main(int argc, char** argv) {
#ifdef AL_HAS_CUDA
  set_device();
  cudaStreamCreate(&bm_stream);
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI";
  if (argc >= 2) {
    backend = argv[1];
  }
  if (argc == 3) {
    max_size = std::atoi(argv[2]);
  }
  if (argc == 4) {
    start_size = std::atoi(argv[2]);
    max_size = std::atoi(argv[3]);
  }

  if (backend == "MPI") {
    do_benchmark<Al::MPIBackend>();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    do_benchmark<Al::NCCLBackend>();
#endif    
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    do_benchmark<Al::MPICUDABackend>();
#endif    
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    do_benchmark<Al::HTBackend>();
#endif
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI";
#ifdef AL_HAS_NCCL
    std::cerr << " | NCCL";
#endif
#ifdef AL_HAS_MPI_CUDA
    std::cerr << " | MPI-CUDA";
#endif
#ifdef AL_HAS_HOST_TRANSFER
    std::cerr << " | HT";
#endif
    std::cerr << "]" << std::endl;
    return -1;
  }

  Al::Finalize();
  return 0;
}
