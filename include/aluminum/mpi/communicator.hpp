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

#include <mpi.h>
#include "aluminum/mpi_comm_and_stream_wrapper.hpp"

namespace Al {
namespace internal {
namespace mpi {

int get_max_tag();

/** Communicator for MPI-based operations. */
class MPICommunicator : public MPICommAndStreamWrapper<int> {
 public:
  /** Default constructor; use MPI_COMM_WORLD. */
  MPICommunicator() : MPICommunicator(MPI_COMM_WORLD) {}
  /**
   * Use a particular MPI communicator and stream.
   *
   * The MPI backend currently ignores streams.
   */
  MPICommunicator(MPI_Comm comm_, int = 0) :
    MPICommAndStreamWrapper<int>(comm_, 0) {}
  /** Cannot copy this. */
  MPICommunicator(const MPICommunicator& other) = delete;
  /** Default move constructor. */
  MPICommunicator(MPICommunicator&& other) = default;
  /** Cannot copy this. */
  MPICommunicator& operator=(MPICommunicator& other) = delete;
  /** Default move assignment operator. */
  MPICommunicator& operator=(MPICommunicator&& other) = default;
  ~MPICommunicator() {}

  /** Create a new MPICommunicator with the same processes. */
  MPICommunicator copy(int stream = 0) const {
    return MPICommunicator(get_comm(), stream);
  }

  /**
   * Return the next free tag on this communicator.
   *
   * TODO: This is meant for internal use and should be moved / eliminted.
   */
  int get_free_tag() {
    int tag = free_tag++;
    if (free_tag >= internal::mpi::get_max_tag()
        || free_tag < starting_free_tag) {
      free_tag = starting_free_tag;
    }
    return tag;
  }

 private:
  /**
   * Starting tag to use for non-blocking operations.
   * No other operations should use any tag >= to this one.
   */
  static constexpr int starting_free_tag = 10;
  /** Free tag for communication. */
  int free_tag = starting_free_tag;
};

} // namespace mpi
} // namespace internal
} // namespace Al
