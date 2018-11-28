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
#include <cmath>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_CUDA
#include "wait.hpp"
#endif
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

size_t start_size = 1;
size_t max_size = 1<<22;
const size_t num_trials = 10;

template <typename Backend>
void do_work(size_t size, typename Backend::comm_type& comm);

template <>
void do_work<Al::MPIBackend>(size_t size, typename Al::MPIBackend::comm_type&) {
  const double sleep_time = 0.0001*std::log2(size);
  std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
}

#ifdef AL_HAS_NCCL
template <>
void do_work<Al::NCCLBackend>(size_t size, typename Al::NCCLBackend::comm_type& comm) {
  const double sleep_time = 0.0001*std::log2(size);
  gpu_wait(sleep_time, comm.get_stream());
}
#endif

#ifdef AL_HAS_MPI_CUDA
template <>
void do_work<Al::MPICUDABackend>(size_t size, typename Al::MPICUDABackend::comm_type& comm) {
  const double sleep_time = 0.0001*std::log2(size);
  gpu_wait(sleep_time, comm.get_stream());
}
#endif

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

// Times *unoverlapped* communication time (roughly).
template <typename Backend>
void time_allreduce_algo(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::allreduce_algo_type algo) {
  std::vector<double> times, in_place_times, work_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);
    typename Backend::req_type req = get_request<Backend>();
    // Estimate work time to compute unoverlapped amount.
    start_timer<Backend>(comm);
    do_work<Backend>(input.size(), comm);
    double work_time = finish_timer<Backend>(comm);
    work_times.push_back(work_time);
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::NonblockingAllreduce<Backend>(
        input.data(), recv.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    do_work<Backend>(input.size(), comm);
    Al::Wait<Backend>(req);
    times.push_back(finish_timer<Backend>(comm) - work_time);
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::NonblockingAllreduce<Backend>(
        in_place_input.data(), input.size(),
        Al::ReductionOperator::sum, comm, req, algo);
    do_work<Backend>(input.size(), comm);
    Al::Wait<Backend>(req);
    in_place_times.push_back(finish_timer<Backend>(comm) - work_time);
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  work_times.erase(work_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " work ";
    print_stats(work_times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " regular ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " inplace ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void do_benchmark() {
  std::vector<typename Backend::allreduce_algo_type> algos
      = get_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm = get_comm<Backend>();
  std::vector<size_t> sizes = {0};
  for (size_t size = start_size; size <= max_size; size *= 2) {
    sizes.push_back(size);
  }
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    // Benchmark algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      time_allreduce_algo<Backend>(data, comm, algo);        
    }
  }
}

int main(int argc, char *argv[]) {
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
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI";
#ifdef AL_HAS_NCCL
    std::cerr << " | NCCL";
#endif
#ifdef AL_HAS_MPI_CUDA
    std::cerr << " | MPI-CUDA";
#endif
    std::cerr << "]" << std::endl;
    return -1;
  }

  Al::Finalize();
  return 0;
}
