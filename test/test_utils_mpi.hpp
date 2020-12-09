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
