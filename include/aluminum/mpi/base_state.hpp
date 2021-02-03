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

#include "aluminum/progress.hpp"

namespace Al {
namespace internal {
namespace mpi {

class MPIState : public AlState {
public:
  MPIState(AlRequest req_) : AlState(req_) {}

  void start() override {
    AlState::start();
    start_mpi_op();
  }

  PEAction step() override {
    if (poll_mpi()) {
      return PEAction::complete;
    } else {
      return PEAction::cont;
    }
  }

protected:
  /** Start the MPI operation and set the request. */
  virtual void start_mpi_op() = 0;
  /** Return the MPI request that will be polled on. */
  MPI_Request* get_mpi_req() { return &mpi_req; }
  /** Return true when the MPI operation is complete. */
  virtual bool poll_mpi() {
    int flag;
    MPI_Test(get_mpi_req(), &flag, MPI_STATUS_IGNORE);
    return flag;
  }

private:
  MPI_Request mpi_req = MPI_REQUEST_NULL;
};

}  // namespace mpi
}  // namespace internal
}  // namespace Al
