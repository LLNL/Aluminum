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

#include "aluminum/cuda/cuda.hpp"
#include "aluminum/ht/communicator.hpp"
#include "aluminum/ht/base_state.hpp"

namespace Al {
namespace internal {
namespace ht {

template <typename T>
class ReduceAlState : public HostTransferCollectiveSignalNonRootEarlyState {
public:
  ReduceAlState(const T* sendbuf, T* recvbuf, size_t count_, ReductionOperator op_,
                int root_, HostTransferCommunicator& comm_, AlGpuStream_t stream_) :
    HostTransferCollectiveSignalNonRootEarlyState(comm_.rank() == root_, stream_),
    host_mem(mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(count_)),
    count(count_),
    root(root_),
    op(mpi::ReductionOperator2MPI_Op<T>(op_)),
    comm(comm_.get_comm()) {
    // Transfer data from device to host.
    AL_CHECK_CUDA(AlGpuMemcpyAsync(host_mem, sendbuf, sizeof(T)*count,
                                  AlGpuMemcpyDeviceToHost, stream_));
    start_event.record(stream_);

    // Have the device wait on the host.
    gpu_wait.wait(stream_);

    if (is_root) {
      // Transfer completed buffer back to device.
      AL_CHECK_CUDA(AlGpuMemcpyAsync(recvbuf, host_mem, sizeof(T)*count,
                                    AlGpuMemcpyHostToDevice, stream_));
    }
    end_event.record(stream_);
  }

  ~ReduceAlState() override {
    mempool.release<MemoryType::CUDA_PINNED_HOST>(host_mem);
  }

  std::string get_name() const override { return "HTReduce"; }

protected:
  void start_mpi_op() override {
    if (is_root) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Ireduce)(
        MPI_IN_PLACE, host_mem, count, mpi::TypeMap<T>(),
        op, root, comm, get_mpi_req());
    } else {
      AL_MPI_LARGE_COUNT_CALL(MPI_Ireduce)(
        host_mem, host_mem, count, mpi::TypeMap<T>(),
        op, root, comm, get_mpi_req());
    }
  }

private:
  T* host_mem;
  size_t count;
  int root;
  MPI_Op op;
  MPI_Comm comm;
};

}  // namespace ht
}  // namespace internal
}  // namespace Al
