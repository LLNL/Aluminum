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
 * Compile-time traits for the Aluminum NCCL backend.
 */

#pragma once

#include "aluminum/traits/traits_base.hpp"
#include "aluminum/nccl_impl.hpp"
#include "traits_base.hpp"

namespace Al {

template <> inline constexpr char AlBackendName<NCCLBackend>[] = "nccl";

// Define the supported NCCL operators.
template <> struct IsOpSupported<AlOperation::allgather, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allgatherv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allreduce, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoall, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoallv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::barrier, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::bcast, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gather, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gatherv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatter, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatterv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatter, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatterv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::send, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::recv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::sendrecv, NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::multisendrecv, NCCLBackend> : std::true_type {};

// Define the types NCCL supports.
template <> struct IsTypeSupported<NCCLBackend, char> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, unsigned char> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, int> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, unsigned int> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, long long int> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, unsigned long long int> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, __half> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, al_bfloat16> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, float> : std::true_type {};
template <> struct IsTypeSupported<NCCLBackend, double> : std::true_type {};

// NCCL supports sum/prod/min/max/avg reductions.
template <>
struct IsReductionOpSupported<NCCLBackend, ReductionOperator::sum> : std::true_type {};
template <>
struct IsReductionOpSupported<NCCLBackend, ReductionOperator::prod> : std::true_type {};
template <>
struct IsReductionOpSupported<NCCLBackend, ReductionOperator::min> : std::true_type {};
template <>
struct IsReductionOpSupported<NCCLBackend, ReductionOperator::max> : std::true_type {};
template <>
struct IsReductionOpSupported<NCCLBackend, ReductionOperator::avg> : std::true_type {};

// Define the algorithm types.
#define AL_ADD_OP_ALGO_TYPE_NCCL(op) AL_ADD_OP_ALGO_TYPE(op, NCCLBackend)
AL_ADD_OP_ALGO_TYPE_NCCL(allgather);
AL_ADD_OP_ALGO_TYPE_NCCL(allgatherv);
AL_ADD_OP_ALGO_TYPE_NCCL(allreduce);
AL_ADD_OP_ALGO_TYPE_NCCL(alltoall);
AL_ADD_OP_ALGO_TYPE_NCCL(alltoallv);
AL_ADD_OP_ALGO_TYPE_NCCL(barrier);
AL_ADD_OP_ALGO_TYPE_NCCL(bcast);
AL_ADD_OP_ALGO_TYPE_NCCL(gather);
AL_ADD_OP_ALGO_TYPE_NCCL(gatherv);
AL_ADD_OP_ALGO_TYPE_NCCL(reduce);
AL_ADD_OP_ALGO_TYPE_NCCL(reduce_scatter);
AL_ADD_OP_ALGO_TYPE_NCCL(reduce_scatterv);
AL_ADD_OP_ALGO_TYPE_NCCL(scatter);
AL_ADD_OP_ALGO_TYPE_NCCL(scatterv);
#undef AL_ADD_OP_ALGO_TYPE_NCCL

}  // namespace Al
