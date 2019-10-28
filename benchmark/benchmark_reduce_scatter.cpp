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
#include "test_utils_host_transfer_cuda.hpp"
#endif

size_t start_size = 1;
size_t max_size = 1<<28;
const size_t num_trials = 20;

template <typename Backend>
void time_reduce_scatter_algo(typename VectorType<Backend>::type input,
                              typename Backend::comm_type& comm,
                              typename Backend::reduce_scatter_algo_type algo,
                              CollectiveProfile<Backend, typename Backend::reduce_scatter_algo_type>& prof) {
  const size_t recv_size = input.size() / comm.size();
  auto recv = get_vector<Backend>(recv_size);
  auto in_place_input(input);
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Reduce_scatter<Backend>(input.data(), recv.data(), recv_size,
                                Al::ReductionOperator::sum, comm, algo);
    if (trial > 0) {  // Skip warmup.
      prof.add_result(comm, input.size(), algo, false,
                      finish_timer<Backend>(comm));
    }
    in_place_input = input;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Reduce_scatter<Backend>(in_place_input.data(), recv_size,
                                Al::ReductionOperator::sum, comm, algo);
    if (trial > 0) {  // Skip warmup.
      prof.add_result(comm, input.size(), algo, true,
                      finish_timer<Backend>(comm));
    }
  }
}

template <typename Backend>
void do_benchmark(const std::vector<size_t> &sizes) {
  std::vector<typename Backend::reduce_scatter_algo_type> algos
      = get_reduce_scatter_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
  CollectiveProfile<Backend, typename Backend::reduce_scatter_algo_type> prof("reduce-scatter");
  for (const auto& size : sizes) {
    const size_t global_size = size * comm.size();
    auto data = gen_data<Backend>(global_size);
    // Benchmark algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      time_reduce_scatter_algo<Backend>(data, comm, algo, prof);
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
  // Add algorithms to test here.

  std::string backend = "MPI";
  if (argc >= 2) {
    backend = argv[1];
  }

  if (argc == 3) {
    start_size = std::atoi(argv[2]);
    max_size = start_size;
  }
  if (argc == 4) {
    start_size = std::atoi(argv[2]);
    max_size = std::atoi(argv[3]);
  }
  std::vector<size_t> sizes = get_sizes(start_size, max_size);
  
  if (backend == "MPI") {
    std::cout << "Reduce-scatter not supported on MPI backend." << std::endl;
    std::abort();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    do_benchmark<Al::NCCLBackend>(sizes);
#endif    
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    do_benchmark<Al::HTBackend>(sizes);
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
