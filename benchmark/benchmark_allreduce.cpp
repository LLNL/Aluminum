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

const size_t num_trials = 20;

template <typename Backend>
void time_allreduce_algo(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::allreduce_algo_type algo,
                         CollectiveProfile<Backend, typename Backend::allreduce_algo_type>& prof) {
  auto recv = get_vector<Backend>(input.size());
  auto in_place_input(input);
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    if (trial > 0) {  // Skip warmup.
      prof.add_result(comm, input.size(), algo, false,
                      finish_timer<Backend>(comm));
    }
    in_place_input = input;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(in_place_input.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    if (trial > 0) {  // Skip warmup.
      prof.add_result(comm, input.size(), algo, true,
                      finish_timer<Backend>(comm));
    }
  }
}

template <typename Backend>
void do_benchmark(const std::vector<size_t> &sizes) {
  std::vector<typename Backend::allreduce_algo_type> algos
      = get_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
  CollectiveProfile<Backend, typename Backend::allreduce_algo_type> prof("allreduce");
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    // Benchmark algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      time_allreduce_algo<Backend>(data, comm, algo, prof);
    }
  }
  if (comm.rank() == 0) {
    prof.print_result_table();
    std::cout << std::flush;
  }
}

int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
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
    std::cout << "Allreduce not supported on MPI-CUDA backend." << std::endl;
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
