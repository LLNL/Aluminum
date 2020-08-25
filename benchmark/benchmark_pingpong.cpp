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
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_ht.hpp"
#endif
#ifdef AL_HAS_CUDA
#include "wait.hpp"
#endif

size_t num_trials = 10000;
size_t num_warmups = 100;

template <typename Backend>
void do_wait(double length, typename Backend::comm_type& comm);

// This is a NOP.
template <>
void do_wait<Al::MPIBackend>(double,
                             typename Al::MPIBackend::comm_type&) {}

#ifdef AL_HAS_NCCL
template <>
void do_wait<Al::NCCLBackend>(
  double length,
  typename Al::NCCLBackend::comm_type& comm) {
  gpu_wait(length, comm.get_stream());
}
#endif

#ifdef AL_HAS_HOST_TRANSFER
template <>
void do_wait<Al::HostTransferBackend>(
  double length, typename Al::HostTransferBackend::comm_type& comm) {
  gpu_wait(length, comm.get_stream());
}
#endif

template <typename Backend>
void time_send_then_recv(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         PtToPtProfile<Backend>& prof) {
  auto recv = get_vector<Backend>(input.size());
  MPI_Barrier(MPI_COMM_WORLD);
  for (size_t trial = 0; trial < num_trials + num_warmups; ++trial) {
    do_wait<Backend>(0.0001, comm);  // To hide launch latency on GPUs.
    start_timer<Backend>(comm);
    if (comm.rank() % 2 == 0) {
      Al::Send<Backend>(input.data(), input.size(), comm.rank() + 1, comm);
      Al::Recv<Backend>(recv.data(), recv.size(), comm.rank() + 1, comm);
    } else {
      Al::Recv<Backend>(recv.data(), recv.size(), comm.rank() - 1, comm);
      Al::Send<Backend>(input.data(), input.size(), comm.rank() - 1, comm);
    }
    if (trial > num_warmups) {
      prof.add_result(comm, input.size(), finish_timer<Backend>(comm) / 2);
    }
  }
}

template <typename Backend>
void time_sendrecv(typename VectorType<Backend>::type input,
                   typename Backend::comm_type& comm,
                   PtToPtProfile<Backend>& prof) {
  auto recv = get_vector<Backend>(input.size());
  MPI_Barrier(MPI_COMM_WORLD);
  for (size_t trial = 0; trial < num_trials + num_warmups; ++trial) {
    do_wait<Backend>(0.0001, comm);  // To hide launch latency on GPUs.
    start_timer<Backend>(comm);
    if (comm.rank() % 2 == 0) {
      Al::SendRecv<Backend>(input.data(), input.size(), comm.rank() + 1,
                            recv.data(), recv.size(), comm.rank() + 1, comm);
    } else {
      Al::SendRecv<Backend>(input.data(), input.size(), comm.rank() - 1,
                            recv.data(), recv.size(), comm.rank() - 1, comm);
    }
    if (trial > num_warmups) {
      prof.add_result(comm, input.size(), finish_timer<Backend>(comm) / 2);
    }
  }
}

template <typename Backend>
void do_benchmark(const std::vector<size_t>& sizes) {
  typename Backend::comm_type comm = get_comm_with_stream<Backend>(MPI_COMM_WORLD);
  if (comm.size() % 2 != 0) {
    std::cerr << "Must use an even number of processes" << std::endl;
    std::abort();
  }
  PtToPtProfile<Backend> prof("SendThenRecv");
  PtToPtProfile<Backend> prof_sr("SendRecv");
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    time_send_then_recv<Backend>(data, comm, prof);
    time_sendrecv<Backend>(data, comm, prof_sr);
  }
  if (comm.rank() == 0) {
    prof.print_result_table();
    prof_sr.print_result_table();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (comm.rank() == 1) {
    prof.print_result_table();
    prof_sr.print_result_table();
  }
  free_comm_with_stream<Backend>(comm);
}

int main(int argc, char** argv) {
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI";
  size_t start_size = 1;
  size_t max_size = 1<<28;
  parse_args(argc, argv, backend, start_size, max_size);
  std::vector<size_t> sizes = get_sizes(start_size, max_size);

  if (backend == "MPI") {
    do_benchmark<Al::MPIBackend>(sizes);
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    do_benchmark<Al::NCCLBackend>(sizes);
#endif    
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    std::cout << "Alltoall not supported on MPI-CUDA backend." << std::endl;
    std::abort();
#endif    
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    do_benchmark<Al::HostTransferBackend>(sizes);
#endif
  }

  Al::Finalize();
  return 0;
}
