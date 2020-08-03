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
#include "cuda_aware_mpi/communicator.hpp"

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
  CUDAAwareMPIState(CUDAAwareMPICommunicator& comm, cudaStream_t stream) :
    AlState(nullptr), compute_stream(comm.get_stream()) {
    pending_ops_event.record(stream);
    gpu_wait.wait(stream);
  }

  PEAction step() override {
    if (!op_ready) {
      // Wait for pending operations.
      if (pending_ops_event.query()) {
        op_ready = true;
        return PEAction::advance;
      } else {
        return PEAction::cont;
      }
    }
    if (!op_started) {
      start_mpi_op();
      op_started = true;
    }
    if (poll_mpi()) {
      gpu_wait.signal();
      return PEAction::complete;
    } else {
      return PEAction::cont;
    }
  }

  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }

 protected:

  /**
   * Child classes should use this to begin an MPI operation and set the
   * MPI request.
   */
  virtual void start_mpi_op() = 0;
  /** Return the MPI request that will be polled on. */
  MPI_Request* get_mpi_req() { return &mpi_req; }
  /**
   * Poll for MPI request completion. Can be overridden for more complex
   * operators.
   */
  virtual bool poll_mpi() {
    int flag;
    MPI_Test(get_mpi_req(), &flag, MPI_STATUS_IGNORE);
    return flag;
  }
 private:
  cudaStream_t compute_stream;
  cuda::FastEvent pending_ops_event;
  cuda::GPUWait gpu_wait;
  bool op_ready = false;
  bool op_started = false;
  MPI_Request mpi_req = MPI_REQUEST_NULL;
};

}  // namespace cuda_aware_mpi
}  // namespace internal
}  // namespace Al
