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
#include <Al_config.hpp>
#include "aluminum/mpi_comm_and_stream_wrapper.hpp"
#include "aluminum/mpi/communicator.hpp"
#include "aluminum/mpi/utils.hpp"

namespace Al {
namespace internal {
namespace ht {

// Define the tag that point-to-point operations will use.
constexpr int pt2pt_tag = 2;

/** Communicator for host-transfer operations. */
class HostTransferCommunicator: public MPICommAndStreamWrapper<AlGpuStream_t> {
 public:
  /** Use Aluminum's world and the default CUDA stream. */
  HostTransferCommunicator()
    : HostTransferCommunicator(mpi::get_world_comm().get_comm(), 0) {}
  /** Use a particular MPI communicator and stream. */
  HostTransferCommunicator(MPI_Comm comm_, AlGpuStream_t stream_ = 0)
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
  HostTransferCommunicator copy(AlGpuStream_t stream = 0) {
    return HostTransferCommunicator(get_comm(), stream);
  }
};

} // namespace ht
} // namespace internal
} // namespace Al
