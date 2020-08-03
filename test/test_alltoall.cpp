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

// Per-process send/recv size.
size_t start_size = 1;
size_t max_size = 1<<30;

/**
 * Test alltoall algo on input, check with expected.
 */
template <typename Backend>
void test_alltoall_algo(const typename VectorType<Backend>::type& expected,
                        typename VectorType<Backend>::type input,
                        typename Backend::comm_type& comm,
                        typename Backend::alltoall_algo_type algo) {
  auto recv = get_vector<Backend>(input.size());
  // Test regular alltoall.
  Al::Alltoall<Backend>(input.data(), recv.data(), input.size() / comm.size(),
                        comm, algo);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular alltoall does not match" <<
        std::endl;
    std::abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // Test in-place alltoall.
  Al::Alltoall<Backend>(input.data(), input.size() / comm.size(), comm, algo);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place alltoall does not match" <<
      std::endl;
    std::abort();
  }
}

/**
 * Test non-blocking alltoall algo on input, check with expected.
 */
template <typename Backend>
void test_nb_alltoall_algo(const typename VectorType<Backend>::type& expected,
                           typename VectorType<Backend>::type input,
                           typename Backend::comm_type& comm,
                           typename Backend::alltoall_algo_type algo) {
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(input.size());
  // Test regular alltoall.
  Al::NonblockingAlltoall<Backend>(input.data(), recv.data(),
                                   input.size() / comm.size(),
                                   comm, req, algo);
  Al::Wait<Backend>(req);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular alltoall does not match" <<
      std::endl;
    std::abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // Test in-place alltoall.
  Al::NonblockingAlltoall<Backend>(input.data(), input.size() / comm.size(),
                                   comm, req, algo);
  Al::Wait<Backend>(req);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place alltoall does not match" <<
      std::endl;
    std::abort();
  }
}

template <typename Backend>
void test_correctness() {
  auto algos = get_alltoall_algorithms<Backend>();
  auto nb_algos = get_nb_alltoall_algorithms<Backend>();
  typename Backend::comm_type comm = get_comm_with_stream<Backend>(MPI_COMM_WORLD);
  // Compute sizes to test.
  std::vector<size_t> sizes = get_sizes(start_size, max_size, true);
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    size_t full_size = size * comm.size();
    // Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(full_size);
    auto expected(data);
    get_expected_alltoall_result(expected);
    // Test algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << Al::algorithm_name(algo) << std::endl;
      }
      test_alltoall_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << Al::algorithm_name(algo) << std::endl;
      }
      test_nb_alltoall_algo<Backend>(expected, data, comm, algo);
    }
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
    std::cerr << "Alltoall not supported on NCCL backend." << std::endl;
    std::abort();
#endif
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    std::cerr << "Alltoall not supported on MPI-CUDA backend." << std::endl;
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
