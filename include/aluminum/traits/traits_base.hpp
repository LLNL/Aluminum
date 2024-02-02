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
 * Base traits for describing Aluminum communication options.
 */

#pragma once

#include <type_traits>

#include "aluminum/base.hpp"
#include "aluminum/datatypes.hpp"

namespace Al {

/** Define Aluminum operations. */
enum class AlOperation {
  allgather,
  allgatherv,
  allreduce,
  alltoall,
  alltoallv,
  barrier,
  bcast,
  gather,
  gatherv,
  reduce,
  reduce_scatter,
  reduce_scatterv,
  scatter,
  scatterv,
  send,
  recv,
  sendrecv,
  multisendrecv
};

/** Give a textual name for each operation. */
template <AlOperation Op> constexpr char AlOperationName[] = "unknown";
template <> inline constexpr char AlOperationName<AlOperation::allgather>[] = "allgather";
template <> inline constexpr char AlOperationName<AlOperation::allgatherv>[] = "allgatherv";
template <> inline constexpr char AlOperationName<AlOperation::allreduce>[] = "allreduce";
template <> inline constexpr char AlOperationName<AlOperation::alltoall>[] = "alltoall";
template <> inline constexpr char AlOperationName<AlOperation::alltoallv>[] = "alltoallv";
template <> inline constexpr char AlOperationName<AlOperation::barrier>[] = "barrier";
template <> inline constexpr char AlOperationName<AlOperation::bcast>[] = "bcast";
template <> inline constexpr char AlOperationName<AlOperation::gather>[] = "gather";
template <> inline constexpr char AlOperationName<AlOperation::gatherv>[] = "gatherv";
template <> inline constexpr char AlOperationName<AlOperation::reduce>[] = "reduce";
template <> inline constexpr char AlOperationName<AlOperation::reduce_scatter>[] = "reduce_scatter";
template <> inline constexpr char AlOperationName<AlOperation::reduce_scatterv>[] = "reduce_scatterv";
template <> inline constexpr char AlOperationName<AlOperation::scatter>[] = "scatter";
template <> inline constexpr char AlOperationName<AlOperation::scatterv>[] = "scatterv";
template <> inline constexpr char AlOperationName<AlOperation::send>[] = "send";
template <> inline constexpr char AlOperationName<AlOperation::recv>[] = "recv";
template <> inline constexpr char AlOperationName<AlOperation::sendrecv>[] = "sendrecv";
template <> inline constexpr char AlOperationName<AlOperation::multisendrecv>[] = "multisendrecv";

/** Give a textual name for each backend. */
template <typename Backend> constexpr char AlBackendName[] = "unknown";

/** Defines whether an Aluminum backend supports an operator. */
template <AlOperation Op, typename Backend> struct IsOpSupported : std::false_type {};

/** Defines whether an Aluminum backend supports a type. */
template <typename Backend, typename T> struct IsTypeSupported : std::false_type {};

/** Defines whether an operator is a reduction. */
template <AlOperation Op> struct IsReductionOp : std::false_type {};
template <> struct IsReductionOp<AlOperation::allreduce> : std::true_type {};
template <> struct IsReductionOp<AlOperation::reduce> : std::true_type {};
template <> struct IsReductionOp<AlOperation::reduce_scatter> : std::true_type {};
template <> struct IsReductionOp<AlOperation::reduce_scatterv> : std::true_type {};

/** Defines whether an operator is a vector operator. */
template <AlOperation Op> struct IsVectorOp : std::false_type {};
template <> struct IsVectorOp<AlOperation::allgatherv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::alltoallv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::gatherv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::reduce_scatterv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::scatterv> : std::true_type {};

/**
 * Defines whether an operator is a collective operation.
 *
 * A collective operation requires every rank in a communicator to
 * participate.
 */
template <AlOperation Op> struct IsCollectiveOp : std::false_type {};
template <> struct IsCollectiveOp<AlOperation::allgather> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::allgatherv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::allreduce> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::alltoall> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::alltoallv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::barrier> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::bcast> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::gather> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::gatherv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::reduce> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::reduce_scatter> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::reduce_scatterv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::scatter> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::scatterv> : std::true_type {};

/**
 * Defines whether an operator is a point-to-point operation.
 *
 * A point-to-point operation involves a single send and/or receive.
 */
template <AlOperation Op> struct IsPt2PtOp : std::false_type {};
template <> struct IsPt2PtOp<AlOperation::send> : std::true_type {};
template <> struct IsPt2PtOp<AlOperation::recv> : std::true_type {};
template <> struct IsPt2PtOp<AlOperation::sendrecv> : std::true_type {};

/**
 * Defines whether an operator has a root.
 *
 * An operator has a root if it distinguishes a special rank in a
 * communicator in some way.
 */
template <AlOperation Op> struct IsRootedOp : std::false_type {};
template <> struct IsRootedOp<AlOperation::bcast> : std::true_type {};
template <> struct IsRootedOp<AlOperation::gather> : std::true_type {};
template <> struct IsRootedOp<AlOperation::gatherv> : std::true_type {};
template <> struct IsRootedOp<AlOperation::reduce> : std::true_type {};
template <> struct IsRootedOp<AlOperation::scatter> : std::true_type {};
template <> struct IsRootedOp<AlOperation::scatterv> : std::true_type {};

/** Defines whether a reduction operator is supported by a backend. */
template <typename Backend, ReductionOperator op>
struct IsReductionOpSupported : std::false_type {};

/** Defines whether an operator can support different algorithms. */
template <AlOperation Op> struct OpSupportsAlgos : std::false_type {};
template <> struct OpSupportsAlgos<AlOperation::allgather> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::allgatherv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::allreduce> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::alltoall> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::alltoallv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::bcast> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::barrier> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::gather> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::gatherv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::reduce> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::reduce_scatter> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::reduce_scatterv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::scatter> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::scatterv> : std::true_type {};

/** Identify the algorithm type for backend's operator. */
template <AlOperation Op, typename Backend> struct OpAlgoType {};

/** Helper for defining simple algorithm type structs. */
#define AL_ADD_OP_ALGO_TYPE(op, backend)                    \
  template <> struct OpAlgoType<AlOperation::op, backend> { \
    using type = backend::op##_algo_type;                   \
  }

/**
 * Define types supported by MPI.
 *
 * Note these are separate from the types the Aluminum MPI backend
 * supports and refer to those supported specifically by MPI itself or
 * by MPI plus extensions provided by Aluminum using native extension
 * means (specifically the way Aluminum adds support for half and
 * bfloat using custom reduction operations).
 */
template <typename T> struct IsTypeSupportedByMPI : std::false_type {};
template <> struct IsTypeSupportedByMPI<char> : std::true_type {};
template <> struct IsTypeSupportedByMPI<signed char> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned char> : std::true_type {};
template <> struct IsTypeSupportedByMPI<short> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned short> : std::true_type {};
template <> struct IsTypeSupportedByMPI<int> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned int> : std::true_type {};
template <> struct IsTypeSupportedByMPI<long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<long long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned long long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<float> : std::true_type {};
template <> struct IsTypeSupportedByMPI<double> : std::true_type {};
template <> struct IsTypeSupportedByMPI<long double> : std::true_type {};
#ifdef AL_HAS_HALF
template <> struct IsTypeSupportedByMPI<__half> : std::true_type {};
#endif
#ifdef AL_HAS_BFLOAT
template <> struct IsTypeSupportedByMPI<al_bfloat16> : std::true_type {};
#endif

}  // namespace Al
