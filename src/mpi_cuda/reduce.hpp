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
class ReduceAlState : public AlState {
public:
  ReduceAlState(const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
                int root, MPICUDACommunicator& comm, cudaStream_t stream) :
    AlState(nullptr),
    rank_(comm.rank()), root_(root), count_(count),
    host_mem_(get_pinned_memory<T>(count_)),
    op_(mpi::ReductionOperator2MPI_Op(op)),
    comm_(comm.get_comm()),
    compute_stream(comm.get_stream()) {

    bool const i_am_root = rank_ == root_;

    // Transfer data from device to host and use an event to determine when it
    // completes.
    AL_CHECK_CUDA(cudaMemcpyAsync(host_mem_, sendbuf, sizeof(T)*count_,
                                  cudaMemcpyDeviceToHost, stream));
    d2h_event_.record(stream);
    gpuwait_.wait(stream);

    if (i_am_root) {
      // Transfer completed buffer back to device.
      AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem_, sizeof(T)*count,
                                    cudaMemcpyHostToDevice, stream));
      h2d_event_.record(stream);
    }
  }

  ~ReduceAlState() override {
    release_pinned_memory(host_mem_);
  }

  bool step() override {
    if (!reduce_started_) {
      // Check if mem xfer complete
      if (d2h_event_.query()) {
        if (root_ == rank_) {
          MPI_Ireduce(MPI_IN_PLACE, host_mem_, count_, mpi::TypeMap<T>(),
                      op_, root_, comm_, &req_);
        } else {
          MPI_Ireduce(host_mem_, host_mem_, count_, mpi::TypeMap<T>(),
                      op_, root_, comm_, &req_);
          gpuwait_.signal();
        }
        reduce_started_ = true;
      }
      else {
        return false;
      }
    }

    if (!reduce_done_) {
      // Wait for the reduce to complete
      int flag;
      MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        reduce_done_ = true;
        if (rank_ == root_) {
          gpuwait_.signal();
        } else {
          return true;
        }
      }
      else {
        return false;
      }
    }

    // Wait for host-to-device memcopy; cleanup
    if (h2d_event_.query()) {
      return true;
    }

    return false;
  }

  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }

private:
  int rank_;
  int root_;
  size_t count_;
  T* host_mem_;

  cuda::GPUWait gpuwait_;

  cuda::FastEvent d2h_event_, h2d_event_;

  MPI_Op op_;
  MPI_Comm comm_;
  MPI_Request req_ = MPI_REQUEST_NULL;

  bool reduce_started_ = false;
  bool reduce_done_ = false;

  cudaStream_t compute_stream;
};

}  // namespace mpi_cuda
}  // namespace internal
}  // namespace Al
