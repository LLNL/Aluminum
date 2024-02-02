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
 * Compile-time traits for the Aluminum host-transfer backend.
 */

#pragma once

#include "aluminum/traits/traits_base.hpp"
#include "aluminum/ht_impl.hpp"
#include "aluminum/datatypes.hpp"

namespace Al {

template <> inline constexpr char AlBackendName<HostTransferBackend>[] = "ht";

// Define the supported host-transfer operators.
template <> struct IsOpSupported<AlOperation::allgather, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allgatherv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allreduce, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoall, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoallv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::barrier, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::bcast, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gather, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gatherv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatter, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatterv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatter, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatterv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::send, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::recv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::sendrecv, HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::multisendrecv, HostTransferBackend> : std::true_type {};

// Define the types host-transfer supports.
template <> struct IsTypeSupported<HostTransferBackend, char> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, signed char> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, unsigned char> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, short> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, unsigned short> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, int> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, unsigned int> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, long> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, unsigned long> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, long long> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, unsigned long long> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, float> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, double> : std::true_type {};
template <> struct IsTypeSupported<HostTransferBackend, long double> : std::true_type {};
#ifdef AL_HAS_HALF
template <> struct IsTypeSupported<HostTransferBackend, __half> : std::true_type {};
#endif
#ifdef AL_HAS_BFLOAT
template <> struct IsTypeSupported<HostTransferBackend, al_bfloat16> : std::true_type {};
#endif

// Host-transfer supports all reduction operators.
template <ReductionOperator op>
struct IsReductionOpSupported<HostTransferBackend, op> : std::true_type {};

// Define the algorithm types.
#define AL_ADD_OP_ALGO_TYPE_HT(op) AL_ADD_OP_ALGO_TYPE(op, HostTransferBackend)
AL_ADD_OP_ALGO_TYPE_HT(allgather);
AL_ADD_OP_ALGO_TYPE_HT(allgatherv);
AL_ADD_OP_ALGO_TYPE_HT(allreduce);
AL_ADD_OP_ALGO_TYPE_HT(alltoall);
AL_ADD_OP_ALGO_TYPE_HT(alltoallv);
AL_ADD_OP_ALGO_TYPE_HT(barrier);
AL_ADD_OP_ALGO_TYPE_HT(bcast);
AL_ADD_OP_ALGO_TYPE_HT(gather);
AL_ADD_OP_ALGO_TYPE_HT(gatherv);
AL_ADD_OP_ALGO_TYPE_HT(reduce);
AL_ADD_OP_ALGO_TYPE_HT(reduce_scatter);
AL_ADD_OP_ALGO_TYPE_HT(reduce_scatterv);
AL_ADD_OP_ALGO_TYPE_HT(scatter);
AL_ADD_OP_ALGO_TYPE_HT(scatterv);
#undef AL_ADD_OP_ALGO_TYPE_HT

}  // namespace Al
