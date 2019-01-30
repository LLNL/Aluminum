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
#include "mpi_impl.hpp"
#include <cassert>

namespace Al {
namespace internal {
namespace host_transfer {

/** Progress engine state for the host-transfer allreduce. */
template <typename T>
class HostTransferState : public AlState {
 public:
  HostTransferState(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, HTCommunicator& comm,
                    cudaStream_t stream, AlRequest req_) :
    AlState(req_),
    host_mem(get_pinned_memory<T>(count)),
    compute_stream(comm.get_stream()) {
    if (count <= 1<<9) {
      host_ar = new mpi::MPIRecursiveDoublingAlState<T>(
        IN_PLACE<T>(), host_mem, count, op, comm, get_free_request());
    } else {
      host_ar = new mpi::MPIRabenseifnerAlState<T>(
        IN_PLACE<T>(), host_mem, count, op, comm, get_free_request());
    }

    // Transfer data from device to host and use an event to determine when it
    // completes. Handle in-place vs non-in-place.
    if (sendbuf != recvbuf) {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem, sendbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
    } else {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem, recvbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
    }
    d2h_event.record(stream);

    // Have the device wait on the host.
    gpu_wait.wait(stream);

    // Transfer completed buffer back to device.
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
    h2d_event.record(stream);
  }

  ~HostTransferState() {
    release_pinned_memory(host_mem);
  }

  bool step() override {
    if (!ar_started) {
      // Wait for memory to get to the host.
      if (d2h_event.query()) {
        host_ar->setup();
        ar_started = true;
      } else {
        return false;
      }
    }
    if (!ar_done) {
      // Wait for the allreduce to complete.
      if (host_ar->step()) {
        ar_done = true;
        delete host_ar;  // TODO: Maybe move this.
        // Mark the sync as done to wake up the device.
        gpu_wait.signal();
      } else {
        return false;
      }
    }
    // Wait for the memcpy back to device to complete so we can clean up.
    if (h2d_event.query()) {
      return true;
    }
    return false;
  }
  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }
 private:
  T* host_mem;
  mpi::MPIAlState<T>* host_ar;
  cuda::FastEvent d2h_event, h2d_event;
  cuda::GPUWait gpu_wait;
  bool ar_started = false;
  bool ar_done = false;
  cudaStream_t compute_stream;
};

} // namespace host_transfer
} // namespace internal
} // namespace Al
