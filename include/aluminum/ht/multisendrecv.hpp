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

#include <vector>
#include <numeric>

#include "aluminum/cuda/cuda.hpp"
#include "aluminum/ht/communicator.hpp"
#include "aluminum/ht/base_state.hpp"

namespace Al {
namespace internal {
namespace ht {

template <typename T>
class MultiSendRecvAlState : public HostTransferCollectiveSignalAtEndState {
public:

  MultiSendRecvAlState(std::vector<const T*> send_buffers,
                       std::vector<size_t> send_counts_,
                       std::vector<int> dests_,
                       std::vector<T*> recv_buffers,
                       std::vector<size_t> recv_counts_,
                       std::vector<int> srcs_,
                       HostTransferCommunicator& comm_,
                       AlGpuStream_t stream_) :
    HostTransferCollectiveSignalAtEndState(stream_),
    inplace(false),
    host_send_buffers(send_buffers.size(), nullptr),
    host_recv_buffers(recv_buffers.size(), nullptr),
    send_counts(send_counts_),
    recv_counts(recv_counts_),
    dests(std::move(dests_)),
    srcs(std::move(srcs_)),
    comm(comm_.get_comm()) {
    size_t total_send_size = std::accumulate(
      send_counts_.begin(), send_counts_.end(), size_t{0});
    size_t total_recv_size = std::accumulate(
      recv_counts_.begin(), recv_counts_.end(), size_t{0});
    host_sendbuf = mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(total_send_size);
    host_recvbuf = mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(total_recv_size);
    // Compute the offsets into the host-side buffer.
    T* tmp_ptr = host_sendbuf;
    for (size_t i = 0; i < send_counts_.size(); ++i) {
      host_send_buffers[i] = tmp_ptr;
      tmp_ptr += send_counts_[i];
    }
    tmp_ptr = host_recvbuf;
    for (size_t i = 0; i < recv_counts_.size(); ++i) {
      host_recv_buffers[i] = tmp_ptr;
      tmp_ptr += recv_counts_[i];
    }
    // Transfer data from the device to the host.
    for (size_t i = 0; i < send_buffers.size(); ++i) {
      AL_CHECK_CUDA(AlGpuMemcpyAsync(host_send_buffers[i],
                                     send_buffers[i],
                                     sizeof(T)*send_counts_[i],
                                     AlGpuMemcpyDeviceToHost, stream_));
    }
    start_event.record(stream_);

    // Have the device wait on the host until communication completes.
    gpu_wait.wait(stream_);

    // Transfer the received data back to the device.
    for (size_t i = 0; i < recv_buffers.size(); ++i) {
      AL_CHECK_CUDA(AlGpuMemcpyAsync(recv_buffers[i],
                                     host_recv_buffers[i],
                                     sizeof(T)*recv_counts_[i],
                                     AlGpuMemcpyHostToDevice, stream_));
    }
    end_event.record(stream_);
  }

  MultiSendRecvAlState(std::vector<T*> buffers,
                       std::vector<size_t> counts,
                       std::vector<int> dests_,
                       std::vector<int> srcs_,
                       HostTransferCommunicator& comm_,
                       AlGpuStream_t stream_) :
    HostTransferCollectiveSignalAtEndState(stream_),
    inplace(true),
    host_send_buffers(buffers.size(), nullptr),
    host_recv_buffers(buffers.size(), nullptr),
    send_counts(counts),
    recv_counts(send_counts),
    dests(std::move(dests_)),
    srcs(std::move(srcs_)),
    comm(comm_.get_comm()) {
    size_t total_size = std::accumulate(
      counts.begin(), counts.end(), size_t{0});
    host_sendbuf = mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(total_size);
    host_recvbuf = mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(total_size);
    // Compute offsets into host-side buffers.
    T* tmp_send_ptr = host_sendbuf;
    T* tmp_recv_ptr = host_recvbuf;
    for (size_t i = 0; i < counts.size(); ++i) {
      host_send_buffers[i] = tmp_send_ptr;
      host_recv_buffers[i] = tmp_recv_ptr;
      tmp_send_ptr += counts[i];
      tmp_recv_ptr += counts[i];
    }

    // Transfer data from the device to the host.
    for (size_t i = 0; i < buffers.size(); ++i) {
      AL_CHECK_CUDA(AlGpuMemcpyAsync(host_send_buffers[i],
                                     buffers[i],
                                     sizeof(T)*counts[i],
                                     AlGpuMemcpyDeviceToHost, stream_));
    }
    start_event.record(stream_);

    // Have the device wait on the host until communication completes.
    gpu_wait.wait(stream_);

    // Transfer the received data back to the device.
    for (size_t i = 0; i < buffers.size(); ++i) {
      AL_CHECK_CUDA(AlGpuMemcpyAsync(buffers[i],
                                     host_recv_buffers[i],
                                     sizeof(T)*counts[i],
                                     AlGpuMemcpyHostToDevice, stream_));
    }
    end_event.record(stream_);
  }

  ~MultiSendRecvAlState() override {
    if (host_sendbuf) {
      mempool.release<MemoryType::CUDA_PINNED_HOST>(host_sendbuf);
      host_sendbuf = nullptr;
    }
    if (host_recvbuf) {
      mempool.release<MemoryType::CUDA_PINNED_HOST>(host_recvbuf);
      host_recvbuf = nullptr;
    }
  }

  std::string get_name() const override { return "HTMultiSendRecv"; }

protected:
  void start_mpi_op() override {
    reqs.resize(host_send_buffers.size() + host_recv_buffers.size());
    for (size_t i = 0; i < host_recv_buffers.size(); ++i) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
        host_recv_buffers[i], recv_counts[i], mpi::TypeMap<T>(),
        srcs[i], pt2pt_tag, comm, &reqs[i]);
    }
    for (size_t i = 0; i < host_send_buffers.size(); ++i) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
        host_send_buffers[i], send_counts[i], mpi::TypeMap<T>(),
        dests[i], pt2pt_tag, comm, &reqs[host_recv_buffers.size() + i]);
    }
  }

  bool poll_mpi() override {
    int flag;
    MPI_Testall(reqs.size(), reqs.data(), &flag, MPI_STATUSES_IGNORE);
    return flag;
  }

private:
  bool inplace;
  T* host_sendbuf = nullptr;
  T* host_recvbuf = nullptr;
  std::vector<T*> host_send_buffers;
  std::vector<T*> host_recv_buffers;
  std::vector<size_t> send_counts;
  std::vector<size_t> recv_counts;
  std::vector<int> dests;
  std::vector<int> srcs;
  MPI_Comm comm;
  std::vector<MPI_Request> reqs;
};

}  // namespace ht
}  // namespace internal
}  // namespace Al
