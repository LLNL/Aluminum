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

#include "aluminum/cuda.hpp"
#include "aluminum/ht/communicator.hpp"
#include "aluminum/ht/base_state.hpp"

namespace Al {
namespace internal {
namespace ht {

template <typename T>
class AllgatherAlState : public HostTransferCollectiveSignalAtEndState {
public:
  AllgatherAlState(const T* sendbuf, T* recvbuf, size_t count_,
                   HostTransferCommunicator& comm_, cudaStream_t stream_) :
    HostTransferCollectiveSignalAtEndState(stream_),
    host_mem(get_pinned_memory<T>(comm_.size()*count_)),
    count(count_),
    comm(comm_.get_comm()) {
    // Transfer data from device to host.
    if (sendbuf == recvbuf) {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem + comm_.rank()*count,
                                    sendbuf + comm_.rank()*count,
                                    sizeof(T)*count, cudaMemcpyDeviceToHost,
                                    stream_));
    } else {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem + comm_.rank()*count,
                                    sendbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream_));
    }
    start_event.record(stream_);

    // Have the device wait on the host.
    gpu_wait.wait(stream_);

    // Transfer completed buffer back to device.
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem,
                                  sizeof(T)*count_*comm_.size(),
                                  cudaMemcpyHostToDevice, stream_));
    end_event.record(stream_);
  }

  ~AllgatherAlState() override {
    release_pinned_memory(host_mem);
  }

  std::string get_name() const override { return "HTAllgather"; }

protected:
  void start_mpi_op() override {
    MPI_Iallgather(MPI_IN_PLACE, count, mpi::TypeMap<T>(),
                   host_mem, count, mpi::TypeMap<T>(), comm, get_mpi_req());
  }

private:
  T* host_mem;
  size_t count;
  MPI_Comm comm;
};

}  // namespace ht
}  // namespace internal
}  // namespace Al
