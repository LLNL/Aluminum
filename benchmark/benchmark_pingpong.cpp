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
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_cuda.hpp"
#include "test_utils_mpi_cuda.hpp"
#include "wait.hpp"
#endif

size_t start_size = 1;
size_t max_size = 1<<18;
//size_t start_size = 256;
//size_t max_size = 256;
size_t num_trials = 10000;

#ifdef AL_HAS_MPI_CUDA

void do_benchmark() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  typename Al::MPICUDABackend::comm_type comm(MPI_COMM_WORLD, stream);
  for (size_t size = start_size; size <= max_size; size *= 2) {
    if (comm.rank() == 0) std::cout << "Benchmarking size " << human_readable_size(size) << std::endl;
    std::vector<double> times, sendrecv_times, host_times;
    std::vector<float> host_sendbuf(size, comm.rank());
    std::vector<float> host_recvbuf(size, 0);
    CUDAVector<float> sendbuf(host_sendbuf);
    CUDAVector<float> recvbuf(host_recvbuf);
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t trial = 0; trial < num_trials; ++trial) {
      // Launch a dummy kernel just to match what the GPU version does.
      gpu_wait(0.0001, stream);
      start_timer<Al::MPIBackend>(comm);
      if (comm.rank() == 0) {
        MPI_Send(host_sendbuf.data(), size, MPI_FLOAT, 1, 1, MPI_COMM_WORLD);
        MPI_Recv(host_recvbuf.data(), size, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else if (comm.rank() == 1) {
        MPI_Recv(host_recvbuf.data(), size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(host_sendbuf.data(), size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
      }
      host_times.push_back(finish_timer<Al::MPIBackend>(comm) / 2);
      if (trial % 4 == 0) {
        cudaStreamSynchronize(stream);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t trial = 0; trial < num_trials; ++trial) {
      gpu_wait(0.0001, stream);
      start_timer<Al::MPICUDABackend>(comm);
      if (comm.rank() == 0) {
        Al::Send<Al::MPICUDABackend>(sendbuf.data(), size, 1, comm);
        Al::Recv<Al::MPICUDABackend>(recvbuf.data(), size, 1, comm);
      } else if (comm.rank() == 1) {
        Al::Recv<Al::MPICUDABackend>(recvbuf.data(), size, 0, comm);
        Al::Send<Al::MPICUDABackend>(sendbuf.data(), size, 0, comm);
      }
      times.push_back(finish_timer<Al::MPICUDABackend>(comm) / 2);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t trial = 0; trial < num_trials; ++trial) {
      gpu_wait(0.0001, stream);
      start_timer<Al::MPICUDABackend>(comm);
      if (comm.rank() == 0) {
        Al::SendRecv<Al::MPICUDABackend>(
          sendbuf.data(), size, 1, recvbuf.data(), size, 1, comm);
      } else if (comm.rank() == 1) {
        Al::SendRecv<Al::MPICUDABackend>(
          sendbuf.data(), size, 0, recvbuf.data(), size, 0, comm);
      }
      sendrecv_times.push_back(finish_timer<Al::MPICUDABackend>(comm) / 2);
    }
    times.erase(times.begin());
    host_times.erase(host_times.begin());
    sendrecv_times.erase(sendrecv_times.begin());
    if (comm.rank() == 0) {
      std::cout << "Rank 0:" << std::endl;
      std::cout << "host ";
      print_stats(host_times);
      std::cout << "mpicuda ";
      print_stats(times);
      std::cout << "mpicuda SR ";
      print_stats(times);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm.rank() == 1) {
      std::cout << "Rank 1:" << std::endl;
      std::cout << "host ";
      print_stats(host_times);
      std::cout << "mpicuda ";
      print_stats(times);
      std::cout << "mpicuda SR ";
      print_stats(times);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  cudaStreamDestroy(stream);
}

#endif  // AL_HAS_MPI_CUDA

int main(int argc, char** argv) {
#ifdef AL_HAS_MPI_CUDA
  set_device();
  Al::Initialize(argc, argv);
  do_benchmark();
  Al::Finalize();
#else
  (void) argc;
  (void) argv;
  std::cout << "MPI-CUDA support required" << std::endl;
#endif
  return 0;
}
