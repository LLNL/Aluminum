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

#pragma once

#include "Al.hpp"
#include "aluminum/traits/traits.hpp"

#include "test_utils.hpp"


// Define default algorithms for each operation.
template <>
struct AlgorithmOptions<Al::MPIBackend> {
  typename Al::MPIBackend::allgather_algo_type allgather_algo =
    Al::MPIBackend::allgather_algo_type::automatic;
  typename Al::MPIBackend::allgatherv_algo_type allgatherv_algo =
    Al::MPIBackend::allgatherv_algo_type::automatic;
  typename Al::MPIBackend::allreduce_algo_type allreduce_algo =
    Al::MPIBackend::allreduce_algo_type::automatic;
  typename Al::MPIBackend::alltoall_algo_type alltoall_algo =
    Al::MPIBackend::alltoall_algo_type::automatic;
  typename Al::MPIBackend::alltoallv_algo_type alltoallv_algo =
    Al::MPIBackend::alltoallv_algo_type::automatic;
  typename Al::MPIBackend::barrier_algo_type barrier_algo =
    Al::MPIBackend::barrier_algo_type::automatic;
  typename Al::MPIBackend::bcast_algo_type bcast_algo =
    Al::MPIBackend::bcast_algo_type::automatic;
  typename Al::MPIBackend::gather_algo_type gather_algo =
    Al::MPIBackend::gather_algo_type::automatic;
  typename Al::MPIBackend::gatherv_algo_type gatherv_algo =
    Al::MPIBackend::gatherv_algo_type::automatic;
  typename Al::MPIBackend::reduce_algo_type reduce_algo =
    Al::MPIBackend::reduce_algo_type::automatic;
  typename Al::MPIBackend::reduce_scatter_algo_type reduce_scatter_algo =
    Al::MPIBackend::reduce_scatter_algo_type::automatic;
  typename Al::MPIBackend::reduce_scatterv_algo_type reduce_scatterv_algo =
    Al::MPIBackend::reduce_scatterv_algo_type::automatic;
  typename Al::MPIBackend::scatter_algo_type scatter_algo =
    Al::MPIBackend::scatter_algo_type::automatic;
  typename Al::MPIBackend::scatterv_algo_type scatterv_algo =
    Al::MPIBackend::scatterv_algo_type::automatic;
};
