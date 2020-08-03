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
#ifdef AL_HAS_CUDA_AWARE_MPI
#include "test_utils_cuda_aware_mpi.hpp"
#endif

#include <stdlib.h>
#include <math.h>
#include <string>

size_t start_size = 1;
size_t max_size = 1<<30;

/** Test send/recv. */
template <typename Backend>
void test_send_and_recv(const typename VectorType<Backend>::type& expected_recv,
                        typename VectorType<Backend>::type& to_send,
                        typename Backend::comm_type& comm) {
  if (comm.size() % 2 != 0 && comm.rank() == comm.size() - 1) {
    // No partner, this rank sits out.
    MPI_Barrier(MPI_COMM_WORLD);  // Participate in barrier below.
    return;
  }
  auto recv = get_vector<Backend>(expected_recv.size());
  if (comm.rank() % 2 == 0) {
    Al::Send<Backend>(to_send.data(), to_send.size(), comm.rank() + 1, comm);
  } else {
    Al::Recv<Backend>(recv.data(), recv.size(), comm.rank() - 1, comm);
    if (!check_vector(expected_recv, recv)) {
      std::cout << comm.rank() << ": recv does not match" << std::endl;
      std::abort();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (comm.rank() % 2 == 0) {
    Al::Recv<Backend>(recv.data(), recv.size(), comm.rank() + 1, comm);
    if (!check_vector(expected_recv, recv)) {
      std::cout << comm.rank() << ": recv does not match" << std::endl;
      std::abort();
    }
  } else {
    Al::Send<Backend>(to_send.data(), to_send.size(), comm.rank() - 1, comm);
  }
}

/** Test non-blocking send/recv. */
template <typename Backend>
void test_nb_send_and_recv(
  const typename VectorType<Backend>::type& expected_recv,
  typename VectorType<Backend>::type& to_send,
  typename Backend::comm_type& comm) {
  if (comm.size() % 2 != 0 && comm.rank() == comm.size() - 1) {
    // No partner, this rank sits out.
    MPI_Barrier(MPI_COMM_WORLD);  // Participate in barrier below.
    return;
  }
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(expected_recv.size());
  if (comm.rank() % 2 == 0) {
    Al::NonblockingSend<Backend>(to_send.data(), to_send.size(),
                                 comm.rank() + 1, comm, req);
  } else {
    Al::NonblockingRecv<Backend>(recv.data(), recv.size(), comm.rank() - 1,
                                 comm, req);
  }
  Al::Wait<Backend>(req);
  if (comm.rank() % 2 != 0) {
    if (!check_vector(expected_recv, recv)) {
      std::cout << comm.rank() << ": recv does not match" << std::endl;
      std::abort();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (comm.rank() % 2 == 0) {
    Al::NonblockingRecv<Backend>(recv.data(), recv.size(), comm.rank() + 1,
                                 comm, req);
  } else {
    Al::NonblockingSend<Backend>(to_send.data(), to_send.size(),
                                 comm.rank() - 1, comm, req);
  }
  Al::Wait<Backend>(req);
  if (comm.rank() % 2 == 0) {
    if (!check_vector(expected_recv, recv)) {
      std::cout << comm.rank() << ": recv does not match" << std::endl;
      std::abort();
    }
  }
}

/** Test sendrecv. */
template <typename Backend>
void test_sendrecv(const typename VectorType<Backend>::type& expected_recv,
                    typename VectorType<Backend>::type& to_send,
                    typename Backend::comm_type& comm) {
  if (comm.size() % 2 != 0 && comm.rank() == comm.size() - 1) {
    // No partner, this rank sits out.
    return;
  }
  auto recv = get_vector<Backend>(expected_recv.size());
  if (comm.rank() % 2 == 0) {
    Al::SendRecv<Backend>(to_send.data(), to_send.size(), comm.rank() + 1,
                          recv.data(), recv.size(), comm.rank() + 1, comm);
  } else {
    Al::SendRecv<Backend>(to_send.data(), to_send.size(), comm.rank() - 1,
                          recv.data(), recv.size(), comm.rank() - 1, comm);
  }
  if (!check_vector(expected_recv, recv)) {
    std::cout << comm.rank() << ": recv does not match" << std::endl;
    std::abort();
  }
}

/** Test non-blocking sendrecv. */
template <typename Backend>
void test_nb_sendrecv(const typename VectorType<Backend>::type& expected_recv,
                      typename VectorType<Backend>::type& to_send,
                      typename Backend::comm_type& comm) {
  if (comm.size() % 2 != 0 && comm.rank() == comm.size() - 1) {
    // No partner, this rank sits out.
    return;
  }
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(expected_recv.size());
  if (comm.rank() % 2 == 0) {
    Al::NonblockingSendRecv<Backend>(to_send.data(), to_send.size(),
                                     comm.rank() + 1, recv.data(), recv.size(),
                                     comm.rank() + 1, comm, req);
  } else {
    Al::NonblockingSendRecv<Backend>(to_send.data(), to_send.size(),
                                     comm.rank() - 1, recv.data(), recv.size(),
                                     comm.rank() - 1, comm, req);
  }
  Al::Wait<Backend>(req);
  if (!check_vector(expected_recv, recv)) {
    std::cout << comm.rank() << ": recv does not match" << std::endl;
    std::abort();
  }
}

template <typename Backend>
void test_correctness() {
  typename Backend::comm_type comm = get_comm_with_stream<Backend>(MPI_COMM_WORLD);
  // Compute sizes to test.
  std::vector<size_t> sizes = get_sizes(start_size, max_size, true);
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    // Compute true value.
    std::vector<float> host_to_send(size, comm.rank());
    std::vector<float> host_to_recv(size, comm.rank() +
                                    (comm.rank() % 2 == 0 ? 1 : -1));
    typename VectorType<Backend>::type to_send(host_to_send);
    typename VectorType<Backend>::type to_recv(host_to_recv);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm.rank() == 0) {
      std::cout << " Send/recv" << std::endl;
    }
    test_send_and_recv<Backend>(to_recv, to_send, comm);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm.rank() == 0) {
      std::cout << " NB Send/recv" << std::endl;
    }
    test_nb_send_and_recv<Backend>(to_recv, to_send, comm);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm.rank() == 0) {
      std::cout << " Sendrecv" << std::endl;
    }
    test_sendrecv<Backend>(to_recv, to_send, comm);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm.rank() == 0) {
      std::cout << " NB Sendrecv" << std::endl;
    }
    test_nb_sendrecv<Backend>(to_recv, to_send, comm);
  }
  free_comm_with_stream<Backend>(comm);
}

int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI";
  parse_args(argc, argv, backend, start_size, max_size);

  if (backend == "MPI") {
    test_correctness<Al::MPIBackend>();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    std::cerr << "Point-to-point not supported on NCCL backend." << std::endl;
    std::abort();
#endif
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    std::cerr << "Point-to-point not supported on MPI-CUDA backend." << std::endl;
    std::abort();
#endif
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    test_correctness<Al::HostTransferBackend>();
#endif
#ifdef AL_HAS_CUDA_AWARE_MPI
  } else if (backend == "CUDA-AWARE-MPI") {
    test_correctness<Al::CUDAAwareMPIBackend>();
#endif
  }

  Al::Finalize();
  return 0;
}
