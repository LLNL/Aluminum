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

#include <memory>
#include "Al.hpp"
#include "aluminum/mpi_comm_and_stream_wrapper.hpp"

namespace Al {
namespace internal {
namespace ht {

/** Communicator for host-transfer operations. */
class HostTransferCommunicator: public MPICommAndStreamWrapper<cudaStream_t> {
 public:
  /** Use MPI_COMM_WORLD and the default CUDA stream. */
  HostTransferCommunicator() : HostTransferCommunicator(MPI_COMM_WORLD, 0) {}
  /** Use a particular MPI communicator and stream. */
  HostTransferCommunicator(MPI_Comm comm_, cudaStream_t stream_)
    : MPICommAndStreamWrapper(comm_, stream_) {}
  /** Cannot copy this. */
  HostTransferCommunicator(const HostTransferCommunicator& other) = delete;
  /** Default move constructor. */
  HostTransferCommunicator(HostTransferCommunicator&& other) = default;
  /** Cannot copy this. */
  HostTransferCommunicator& operator=(const HostTransferCommunicator& other) = delete;
  /** Default move assignment operator. */
  HostTransferCommunicator& operator=(HostTransferCommunicator&& other) = default;
  ~HostTransferCommunicator() {}

  /**
   * Create a new HostTransfer communicator with the same processes
   * and a new stream.
   */
  HostTransferCommunicator copy(cudaStream_t stream = 0) {
    return HostTransferCommunicator(get_comm(), stream);
  }
};

} // namespace ht
} // namespace internal
} // namespace Al
