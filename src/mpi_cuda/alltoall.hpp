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
#include "cuda_kernels.hpp"
#include "mpi_cuda/communicator.hpp"
#include "progress.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

template <typename T>
class AlltoallAlState : public AlState {
public:
  AlltoallAlState(const T* sendbuf, T* recvbuf, size_t count,
                  MPICUDACommunicator& comm, cudaStream_t stream) :
    AlState(nullptr),
    host_mem_(get_pinned_memory<T>(comm.size()*count)),
    count_(count),
    sync_(get_pinned_memory<int32_t>(1)),
    sync_dev_ptr_no_mem_ops_(nullptr),
    sync_dev_ptr_(0U),
    d2h_event_(cuda::get_cuda_event()),
    h2d_event_(cuda::get_cuda_event()),
    comm_(comm.get_comm()) {

    bool const use_stream_ops = cuda::stream_memory_operations_supported();

    // Setup the watched memory
    *sync_ = 0;

    if (use_stream_ops)
      AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(&sync_dev_ptr_, sync_, 0));
    else
      AL_CHECK_CUDA(cudaHostGetDevicePointer(&sync_dev_ptr_no_mem_ops_, sync_, 0));

    // Transfer data from device to host and use an event to determine when it
    // completes.
    AL_CHECK_CUDA(cudaMemcpyAsync(host_mem_,sendbuf, sizeof(T)*count*comm.size(),
                                  cudaMemcpyDeviceToHost, stream));
    AL_CHECK_CUDA(cudaEventRecord(d2h_event_, stream));

    // Have the device wait on the host.
    if (use_stream_ops)
      launch_wait_kernel(stream, 1, sync_dev_ptr_);
    else
      launch_wait_kernel(stream, 1, sync_dev_ptr_no_mem_ops_);

    // Transfer completed buffer back to device.
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem_, sizeof(T)*count*comm.size(),
                                  cudaMemcpyHostToDevice, stream));
    AL_CHECK_CUDA(cudaEventRecord(h2d_event_, stream));

  }

  ~AlltoallAlState() override {
    release_pinned_memory(host_mem_);
    release_pinned_memory(sync_);
    cuda::release_cuda_event(h2d_event_);
    cuda::release_cuda_event(d2h_event_);
  }

  bool step() override {
    if (!a2a_started_) {
      // Check if mem xfer complete
      cudaError_t r = cudaEventQuery(d2h_event_);
      if (r == cudaSuccess) {
        MPI_Ialltoall(MPI_IN_PLACE, count_, mpi::TypeMap<T>(),
                      host_mem_, count_, mpi::TypeMap<T>(), comm_, &req_);
        a2a_started_ = true;
      }
      else if (r == cudaErrorNotReady) {
        return false;
      }
      else {
        throw_al_exception("Alltoall: cudaEventQuery error");
      }
    }

    if (!a2a_done_) {
      // Wait for the all2all to complete
      int flag;
      MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        a2a_done_ = true;
        *sync_ = 1; // Mark the sync as done to wake device
      }
      else {
        return false;
      }
    }

    // Wait for host-to-device memcopy; cleanup
    cudaError_t r = cudaEventQuery(h2d_event_);
    if (r == cudaSuccess) {
      return true;
    }
    else if (r != cudaErrorNotReady) {
      throw_al_exception("Alltoall: cudaEventQuery error");
    }

    return false;
  }

  bool needs_completion() const override { return false; }

private:
  T* host_mem_;
  size_t count_;

  int32_t* sync_, * sync_dev_ptr_no_mem_ops_;
  CUdeviceptr sync_dev_ptr_;
  cudaEvent_t d2h_event_, h2d_event_;

  MPI_Comm comm_;
  MPI_Request req_ = MPI_REQUEST_NULL;

  bool a2a_started_ = false;
  bool a2a_done_ = false;
};

}  // namespace mpi_cuda
}  // namespace internal
}  // namespace Al
