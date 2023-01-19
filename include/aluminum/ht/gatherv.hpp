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
class GathervAlState : public HostTransferCollectiveSignalNonRootEarlyState {
public:
  GathervAlState(const T* sendbuf, T* recvbuf,
                 std::vector<size_t> counts_, std::vector<size_t> displs_,
                 int root_,
                 HostTransferCommunicator& comm_, AL_GPU_RT(Stream_t) stream_) :
    HostTransferCollectiveSignalNonRootEarlyState(comm_.rank() == root_, stream_),
    host_mem(mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>((comm_.rank() == root_) ?
                                  (displs_.back() + counts_.back()) :
                                  counts_[comm_.rank()])),
    send_count(counts_[comm_.rank()]),
    counts(mpi::intify_size_t_vector(counts_)),
    displs(mpi::intify_size_t_vector(displs_)),
    root(root_),
    comm(comm_.get_comm()) {
    // Transfer data from device to host.
    if (is_root) {
      AL_CHECK_CUDA(AL_GPU_RT(MemcpyAsync)(
                      host_mem + displs_[comm_.rank()],
                      (sendbuf == recvbuf) ? sendbuf + displs_[comm_.rank()] : sendbuf,
                      sizeof(T) * counts_[comm_.rank()],
                      AL_GPU_RT(MemcpyDeviceToHost), stream_));
    } else {
      AL_CHECK_CUDA(AL_GPU_RT(MemcpyAsync)(host_mem, sendbuf,
                                    sizeof(T) * counts_[comm_.rank()],
                                    AL_GPU_RT(MemcpyDeviceToHost), stream_));
    }
    start_event.record(stream_);

    // Have the device wait on the host.
    gpu_wait.wait(stream_);

    if (is_root) {
      // Transfer completed buffer back to device.
      AL_CHECK_CUDA(AL_GPU_RT(MemcpyAsync)(recvbuf, host_mem,
                                    sizeof(T) * (displs_.back() + counts_.back()),
                                    AL_GPU_RT(MemcpyHostToDevice), stream_));
    }
    end_event.record(stream_);
  }

  ~GathervAlState() override {
    mempool.release<MemoryType::CUDA_PINNED_HOST>(host_mem);
  }

  std::string get_name() const override { return "HTGatherv"; }

protected:
  void start_mpi_op() override {
    if (is_root) {
      MPI_Igatherv(MPI_IN_PLACE, send_count, mpi::TypeMap<T>(),
                   host_mem, counts.data(), displs.data(), mpi::TypeMap<T>(),
                   root, comm, get_mpi_req());
    } else {
      MPI_Igatherv(host_mem, send_count, mpi::TypeMap<T>(),
                   host_mem, counts.data(), displs.data(), mpi::TypeMap<T>(),
                   root, comm, get_mpi_req());
    }
  }

private:
  T* host_mem;
  size_t send_count;
  std::vector<int> counts;
  std::vector<int> displs;
  int root;
  MPI_Comm comm;
};

}  // namespace ht
}  // namespace internal
}  // namespace Al
