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
#include "mpi_cuda/communicator.hpp"
#include "mpi_cuda/allreduce_ring.hpp"
#include "mpi_cuda/util.hpp"
#include "mpi_impl.hpp"
#include <cassert>

namespace Al {
namespace internal {
namespace mpi_cuda {

template <typename T> inline
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, MPICUDACommunicator& comm,
                    cudaStream_t stream) {
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  comm.get_ring().allreduce<T>(recvbuf, count, op, stream, false);
}

template <typename T> inline
void bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm,
                       cudaStream_t stream) {
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  comm.get_ring().allreduce<T>(recvbuf, count, op, stream, true);
}

/** Progress engine state for the host-transfer allreduce. */
template <typename T>
class HostTransferState : public AlState {
 public:
  HostTransferState(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, MPICUDACommunicator& comm,
                    cudaStream_t stream, AlRequest req_) : AlState(req_) {
    host_mem = get_pinned_memory<T>(count);
    if (count <= 1<<9) {
      host_ar = new mpi::MPIRecursiveDoublingAlState<T>(
        IN_PLACE<T>(), host_mem, count, op, comm, get_free_request());
    } else {
      host_ar = new mpi::MPIRabenseifnerAlState<T>(
        IN_PLACE<T>(), host_mem, count, op, comm, get_free_request());
    }
    sync_event = cuda::get_cuda_event();
    sync_event2 = cuda::get_cuda_event();

    // Transfer data from device to host and use an event to determine when it
    // completes. Handle in-place vs non-in-place.
    if (sendbuf != recvbuf) {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem, sendbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
    } else {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem, recvbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
    }
    AL_CHECK_CUDA(cudaEventRecord(sync_event, stream));
    // Have the device wait on the host.

    gpu_wait.wait(stream);

    // Transfer completed buffer back to device.
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
    AL_CHECK_CUDA(cudaEventRecord(sync_event2, stream));
  }

  ~HostTransferState() {

  }

  bool step() override {
    if (!ar_started) {
      // Wait for memory to get to the host.
      cudaError_t r = cudaEventQuery(sync_event);
      if (r == cudaSuccess) {
        host_ar->setup();
        ar_started = true;
      } else if (r == cudaErrorNotReady) {
        return false;
      } else {
        throw_al_exception("cudaEventQuery error");
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
    cudaError_t r = cudaEventQuery(sync_event2);
    if (r == cudaSuccess) {
      release_pinned_memory(host_mem);  // TODO: Maybe move this.
      release_pinned_memory(sync);
      cuda::release_cuda_event(sync_event);
      cuda::release_cuda_event(sync_event2);
      return true;
    } else if (r != cudaErrorNotReady) {
      throw_al_exception("cudaEventQuery error");
    }
    return false;
  }
  bool needs_completion() const override { return false; }
 private:
  cudaEvent_t sync_event;
  cudaEvent_t sync_event2;
  bool ar_started = false;
  bool ar_done = false;
  mpi::MPIAlState<T>* host_ar;
  T* host_mem;
  cuda::GPUWait gpu_wait;
};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
