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

#include "test_utils.hpp"


// Operator support.
template <> struct IsOpSupported<AlOperation::allgather, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allgatherv, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allreduce, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoall, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoallv, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::barrier, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::bcast, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gather, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gatherv, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatter, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatterv, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatter, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatterv, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::send, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::recv, Al::MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::sendrecv, Al::MPIBackend> : std::true_type {};

// Type support.
template <> struct IsTypeSupported<Al::MPIBackend, char> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, signed char> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, unsigned char> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, short> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, unsigned short> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, int> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, unsigned int> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, long> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, unsigned long> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, long long> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, unsigned long long> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, float> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, double> : std::true_type {};
template <> struct IsTypeSupported<Al::MPIBackend, long double> : std::true_type {};

// Reduction operator support (all are supported).
template <Al::ReductionOperator op>
struct IsReductionOpSupported<Al::MPIBackend, op> : std::true_type {};

// Backend name.
template <> constexpr char AlBackendName<Al::MPIBackend>[] = "mpi";

// Algorithm types.
template <> struct OpAlgoType<AlOperation::allgather, Al::MPIBackend> {
  using type = Al::MPIBackend::allgather_algo_type;
};
template <> struct OpAlgoType<AlOperation::allgatherv, Al::MPIBackend> {
  using type = Al::MPIBackend::allgatherv_algo_type;
};
template <> struct OpAlgoType<AlOperation::allreduce, Al::MPIBackend> {
  using type = Al::MPIBackend::allreduce_algo_type;
};
template <> struct OpAlgoType<AlOperation::alltoall, Al::MPIBackend> {
  using type = Al::MPIBackend::alltoall_algo_type;
};
template <> struct OpAlgoType<AlOperation::alltoallv, Al::MPIBackend> {
  using type = Al::MPIBackend::alltoallv_algo_type;
};
template <> struct OpAlgoType<AlOperation::barrier, Al::MPIBackend> {
  using type = Al::MPIBackend::barrier_algo_type;
};
template <> struct OpAlgoType<AlOperation::bcast, Al::MPIBackend> {
  using type = Al::MPIBackend::bcast_algo_type;
};
template <> struct OpAlgoType<AlOperation::gather, Al::MPIBackend> {
  using type = Al::MPIBackend::gather_algo_type;
};
template <> struct OpAlgoType<AlOperation::gatherv, Al::MPIBackend> {
  using type = Al::MPIBackend::gatherv_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce, Al::MPIBackend> {
  using type = Al::MPIBackend::reduce_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce_scatter, Al::MPIBackend> {
  using type = Al::MPIBackend::reduce_scatter_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce_scatterv, Al::MPIBackend> {
  using type = Al::MPIBackend::reduce_scatterv_algo_type;
};
template <> struct OpAlgoType<AlOperation::scatter, Al::MPIBackend> {
  using type = Al::MPIBackend::scatter_algo_type;
};
template <> struct OpAlgoType<AlOperation::scatterv, Al::MPIBackend> {
  using type = Al::MPIBackend::scatterv_algo_type;
};

// Supported algorithms.
template <>
std::vector<std::pair<std::string, typename OpAlgoType<AlOperation::allreduce, Al::MPIBackend>::type>> get_supported_algos<AlOperation::allreduce, Al::MPIBackend>() {
  using algo_type = Al::MPIBackend::allreduce_algo_type;
  return {{"automatic", algo_type::automatic},
          {"passthrough", algo_type::mpi_passthrough},
          {"recursive_doubling", algo_type::mpi_recursive_doubling},
          {"ring", algo_type::mpi_ring},
          {"rabenseifner", algo_type::mpi_rabenseifner},
          {"biring", algo_type::mpi_biring}};
}

// Algorithms.
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
