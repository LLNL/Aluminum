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

/**
 * @file
 * Compile-time traits for the Aluminum MPI backend.
 */

#pragma once

#include "aluminum/traits/traits_base.hpp"
#include "aluminum/datatypes.hpp"
#include "aluminum/mpi_impl.hpp"
#include "traits_base.hpp"

namespace Al {

template <> inline constexpr char AlBackendName<MPIBackend>[] = "mpi";

// Define the supported MPI operators.
template <> struct IsOpSupported<AlOperation::allgather, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allgatherv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allreduce, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoall, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoallv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::barrier, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::bcast, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gather, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gatherv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatter, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatterv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatter, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatterv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::send, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::recv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::sendrecv, MPIBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::multisendrecv, MPIBackend> : std::true_type {};

// Define the types MPI supports.
template <> struct IsTypeSupported<MPIBackend, char> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, signed char> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, unsigned char> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, short> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, unsigned short> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, int> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, unsigned int> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, long> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, unsigned long> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, long long> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, unsigned long long> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, float> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, double> : std::true_type {};
template <> struct IsTypeSupported<MPIBackend, long double> : std::true_type {};
#ifdef AL_HAS_HALF
template <> struct IsTypeSupported<MPIBackend, __half> : std::true_type {};
#endif
#ifdef AL_HAS_BFLOAT
template <> struct IsTypeSupported<MPIBackend, al_bfloat16> : std::true_type {};
#endif

// MPI supports all reduction operators.
template <ReductionOperator op>
struct IsReductionOpSupported<MPIBackend, op> : std::true_type {};

// Define the algorithm types.
#define AL_ADD_OP_ALGO_TYPE_MPI(op) AL_ADD_OP_ALGO_TYPE(op, MPIBackend)
AL_ADD_OP_ALGO_TYPE_MPI(allgather);
AL_ADD_OP_ALGO_TYPE_MPI(allgatherv);
AL_ADD_OP_ALGO_TYPE_MPI(allreduce);
AL_ADD_OP_ALGO_TYPE_MPI(alltoall);
AL_ADD_OP_ALGO_TYPE_MPI(alltoallv);
AL_ADD_OP_ALGO_TYPE_MPI(barrier);
AL_ADD_OP_ALGO_TYPE_MPI(bcast);
AL_ADD_OP_ALGO_TYPE_MPI(gather);
AL_ADD_OP_ALGO_TYPE_MPI(gatherv);
AL_ADD_OP_ALGO_TYPE_MPI(reduce);
AL_ADD_OP_ALGO_TYPE_MPI(reduce_scatter);
AL_ADD_OP_ALGO_TYPE_MPI(reduce_scatterv);
AL_ADD_OP_ALGO_TYPE_MPI(scatter);
AL_ADD_OP_ALGO_TYPE_MPI(scatterv);
#undef AL_ADD_OP_ALGO_TYPE_MPI

}  // namespace Al
