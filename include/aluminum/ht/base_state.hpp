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
#include "aluminum/cuda.hpp"

namespace Al {
namespace internal {
namespace ht {

/** Base implementation for common host-transfer collectives. */
class HostTransferCollectiveState : public AlState {
public:
  HostTransferCollectiveState(cudaStream_t stream_) :
    AlState(nullptr), stream(stream_) {}

  bool needs_completion() const override { return false; }
  void *get_compute_stream() const override { return stream; }

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

  // These are protected to simplify child implementations.

  /** Event that when complete indicates communication can begin. */
  cuda::FastEvent start_event;
  /** Whether start_event has completed. */
  bool start_done = false;
  /** Event that when complete indicates all device operations are done. */
  cuda::FastEvent end_event;
  /** Whether the MPI operation has been started. */
  bool mpi_started = false;
  /** Whether the MPI operation has completed. */
  bool mpi_done = false;
  /** GPU-side wait. */
  cuda::GPUWait gpu_wait;

private:
  /** Associated compute stream. */
  cudaStream_t stream;
  /** Internal MPI request handle. */
  MPI_Request mpi_req = MPI_REQUEST_NULL;
};

/**
 * This implements a step method that signals the GPU wait only after
 * communication has completed.
 */
class HostTransferCollectiveSignalAtEndState : public HostTransferCollectiveState {
public:
  HostTransferCollectiveSignalAtEndState(cudaStream_t stream_) :
    HostTransferCollectiveState(stream_) {}

  PEAction step() override {
    if (!start_done) {
      if (start_event.query()) {
        start_done = true;
        return PEAction::advance;
      } else {
        return PEAction::cont;
      }
    }
    if (!mpi_started) {
      start_mpi_op();
      mpi_started = true;
    }
    if (!mpi_done) {
      if (poll_mpi()) {
        mpi_done = true;
        gpu_wait.signal();
      } else {
        return PEAction::cont;
      }
    }
    if (end_event.query()) {
      return PEAction::complete;
    }
    return PEAction::cont;
  }
};

/**
 * This implements a step method that signals the GPU wait after the
 * start event completes on the root, and after communication has
 * completed elsewhere.
 */
class HostTransferCollectiveSignalRootEarlyState : public HostTransferCollectiveState {
 public:
  HostTransferCollectiveSignalRootEarlyState(bool is_root_, cudaStream_t stream_) :
    HostTransferCollectiveState(stream_), is_root(is_root_) {}

  PEAction step() override {
    if (!start_done) {
      if (start_event.query()) {
        start_done = true;
        return PEAction::advance;
      } else {
        return PEAction::cont;
      }
    }
    if (!mpi_started) {
      start_mpi_op();
      mpi_started = true;
      if (is_root) {
        gpu_wait.signal();
      }
    }
    if (!mpi_done) {
      if (poll_mpi()) {
        mpi_done = true;
        if (!is_root) {
          gpu_wait.signal();
        } else {
          return PEAction::complete;
        }
      } else {
        return PEAction::cont;
      }
    }
    if (end_event.query()) {
      return PEAction::complete;
    }
    return PEAction::cont;
  }

 protected:
  bool is_root;
};

/**
 * This implements a step method that signals the GPU wait after the
 * start event completes on non-root processes, and after communication
 * has completed on the root.
 */
class HostTransferCollectiveSignalNonRootEarlyState : public HostTransferCollectiveState {
 public:
  HostTransferCollectiveSignalNonRootEarlyState(bool is_root_, cudaStream_t stream_) :
    HostTransferCollectiveState(stream_), is_root(is_root_) {}

  PEAction step() override {
    if (!start_done) {
      if (start_event.query()) {
        start_done = true;
        return PEAction::advance;
      } else {
        return PEAction::cont;
      }
    }
    if (!mpi_started) {
      start_mpi_op();
      mpi_started = true;
      if (!is_root) {
        gpu_wait.signal();
      }
    }
    if (!mpi_done) {
      if (poll_mpi()) {
        mpi_done = true;
        if (is_root) {
          gpu_wait.signal();
        } else {
          return PEAction::complete;
        }
      } else {
        return PEAction::cont;
      }
    }
    if (end_event.query()) {
      return PEAction::complete;
    }
    return PEAction::cont;
  }

 protected:
  bool is_root;
};

}  // namespace ht
}  // namespace internal
}  // namespace Al
