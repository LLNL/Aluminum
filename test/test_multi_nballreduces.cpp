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
#include <string>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_host_transfer_cuda.hpp"
#endif

size_t max_size = 1<<20;
const size_t num_concurrent = 1024;
const size_t num_blocking = 8;

template <typename Backend>
void test_multiple_nballreduces() {
  auto algos = get_nb_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
  std::vector<size_t> sizes = {0};
  for (size_t size = 1; size <= max_size; size *= 2) {
    sizes.push_back(size);
    if (size > 1) {
      sizes.push_back(size + 1);
    }
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    for (auto&& algo : algos) {
      std::vector<typename Backend::req_type> reqs(num_concurrent);
      std::vector<typename VectorType<Backend>::type> input_data;
      std::vector<typename VectorType<Backend>::type> expected_results;
      for (size_t i = 0; i < num_concurrent; ++i) {
        input_data.push_back(std::move(gen_data<Backend>(size)));
        expected_results.push_back(std::move(typename VectorType<Backend>::type(input_data[i])));
        get_expected_allreduce_result(expected_results[i]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << Al::algorithm_name(algo) << std::endl;
      }
      // Start each allreduce.
      for (size_t i = 0; i < num_concurrent; ++i) {
        Al::NonblockingAllreduce<Backend>(input_data[i].data(),
                                          input_data[i].size(),
                                          Al::ReductionOperator::sum,
                                          comm,
                                          reqs[i],
                                          algo);
      }
      // This is commented out because I don't have a good way to generalize it,
      // but it can be used to reveal bugs in the MPI-CUDA host-transfer
      // allreduce algorithm (see issue #40).
      // Run dummy blocking allreduces concurrently. Don't check them.
      /*typename VectorType<Al::MPIBackend>::type&& blocking_data = gen_data<Al::MPIBackend>(size);
      for (size_t i = 0; i < num_blocking; ++i) {
        Al::Allreduce<Al::MPIBackend>(blocking_data.data(), blocking_data.size(),
                                      Al::ReductionOperator::sum, comm);
      }*/
      (void) num_blocking;
      // Complete and check them.
      for (size_t i = 0; i < num_concurrent; ++i) {
        Al::Wait<Backend>(reqs[i]);
        if (!check_vector(expected_results[i], input_data[i])) {
          std::cout << comm.rank() << ": allreduce does not match" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI";
  if (argc >= 2) {
    backend = argv[1];
  }
  if (argc == 3) {
    max_size = std::stoul(argv[2]);
  }

  if (backend == "MPI") {
    test_multiple_nballreduces<Al::MPIBackend>();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    test_multiple_nballreduces<Al::NCCLBackend>();
#endif
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    test_multiple_nballreduces<Al::HTBackend>();
#endif
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI";
#ifdef AL_HAS_NCCL
    std::cerr << " | NCCL";
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
