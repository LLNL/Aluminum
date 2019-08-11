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

#include "cuda.hpp"
#include "progress.hpp"
#include "cuda_aware_mpi_impl.hpp"
#include "mpi_impl.hpp"
#include "communicator.hpp"

namespace Al {
namespace internal {
namespace cuda_aware_mpi {

/**
 * Base class for CUDA-aware MPI progress engine states.
 *
 * This handles the basic pattern of waiting for existing operations to
 * complete and polling on an MPI request while blocking the GPU.
 */
class CUDAAwareMPIState : public AlState {
 public:
  CUDAAwareMPIState(cudaStream_t stream) :
    AlState(nullptr) {
    pending_ops_event.record(stream);
    gpu_wait.wait(stream);
  }

  bool step() override {
    if (!op_ready) {
      // Wait for pending operations.
      if (pending_ops_event.query()) {
        op_ready = true;
      }
      // Always return false here to ensure operations start in the right
      // order.
      return false;
    }
    if (!op_started) {
      start_mpi_op();
      op_started = true;
    }
    int flag;
    MPI_Test(get_mpi_req(), &flag, MPI_STATUS_IGNORE);
    if (flag) {
      gpu_wait.signal();
      return true;
    } else {
      return false;
    }
  }

 protected:

  virtual void start_mpi_op() = 0;
  MPI_Request* get_mpi_req() { return &mpi_req; }
 private:
  cuda::FastEvent pending_ops_event;
  cuda::GPUWait gpu_wait;
  bool op_ready = false;
  bool op_started = false;
  MPI_Request mpi_req = MPI_REQUEST_NULL;
};

}  // namespace cuda_aware_mpi
}  // namespace internal
}  // namespace Al
