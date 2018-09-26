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
class ScatterAlState : public AlState {
public:
  ScatterAlState(const T* sendbuf, T* recvbuf, size_t count, int root,
                MPICUDACommunicator& comm, cudaStream_t stream) :
    AlState(nullptr),
    rank_(comm.rank()), root_(root), count_(count),
    host_mem_(get_pinned_memory<T>(rank_ == root_
                                   ? comm.size()*count_ : count_)),
    d2h_event_(rank_ == root_ ? cuda::get_cuda_event() : nullptr),
    h2d_event_(cuda::get_cuda_event()),
    comm_(comm.get_comm()) {

    bool const i_am_root = rank_ == root_;
    bool const inplace_operation = sendbuf == recvbuf;

    // Transfer data from device to host and use an event to determine when it
    // completes.
    if (i_am_root) {
      AL_CHECK_CUDA(cudaMemcpyAsync(
                      host_mem_, sendbuf+count_*comm.size(),
                      sizeof(T)*count_, cudaMemcpyDeviceToHost, stream));
      AL_CHECK_CUDA(cudaEventRecord(d2h_event_, stream));

      // Copy to self
      if (!inplace_operation) {
        AL_CHECK_CUDA(cudaMemcpyAsync(
                        recvbuf, sendbuf + rank_*count_,
                        count_*sizeof(T), cudaMemcpyDeviceToDevice, stream));
      }
    }
    else {
      // Enqueue the kernel to wait on the host; root only
      gpuwait_.wait(stream);

      // Transfer completed buffer back to device.
      AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem_, sizeof(T)*count,
                                    cudaMemcpyHostToDevice, stream));

      AL_CHECK_CUDA(cudaEventRecord(h2d_event_, stream));
    }
  }

  ~ScatterAlState() override {
    release_pinned_memory(host_mem_);
    cuda::release_cuda_event(h2d_event_);
    if (d2h_event_) {
      cuda::release_cuda_event(d2h_event_);
    }
  }

  bool step() override {
    if (!scatter_started_) {
      // Check if mem xfer complete
      cudaError_t r = cudaEventQuery(d2h_event_);
      if (r == cudaSuccess) {
        if (root_ == rank_)
          MPI_Iscatter(host_mem_, count_, mpi::TypeMap<T>(),
                       MPI_IN_PLACE, count_, mpi::TypeMap<T>(),
                       root_, comm_, &req_);
        else
          MPI_Iscatter(host_mem_, count_, mpi::TypeMap<T>(),
                       host_mem_, count_, mpi::TypeMap<T>(),
                       root_, comm_, &req_);
        scatter_started_ = true;
      }
      else if (r == cudaErrorNotReady) {
        return false;
      }
      else {
        throw_al_exception("Alltoall: cudaEventQuery error");
      }
    }

    if (!scatter_done_) {
      // Wait for the all2all to complete
      int flag;
      MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        scatter_done_ = true;
        if (rank_ == root_)
          gpuwait_.signal();
        else
          return true;
      }
      else {
        return false;
      }
    }
    else if (rank_ != root_) {
      // Paranoia, in case step() is ever called again after returning
      // 'true' for the first time.
      return true;
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
  int rank_;
  int root_;
  size_t count_;
  T* host_mem_;

  cuda::GPUWait gpuwait_;

  cudaEvent_t d2h_event_, h2d_event_;

  MPI_Comm comm_;
  MPI_Request req_ = MPI_REQUEST_NULL;

  bool scatter_started_ = false;
  bool scatter_done_ = false;
};

}  // namespace mpi_cuda
}  // namespace internal
}  // namespace Al