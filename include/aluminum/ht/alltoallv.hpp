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
class AlltoallvAlState : public HostTransferCollectiveSignalAtEndState {
public:
  AlltoallvAlState(const T* sendbuf,
                   std::vector<size_t> send_counts_,
                   std::vector<size_t> send_displs_,
                   T* recvbuf,
                   std::vector<size_t> recv_counts_,
                   std::vector<size_t> recv_displs_,
                   HostTransferCommunicator& comm_, cudaStream_t stream_) :
    HostTransferCollectiveSignalAtEndState(stream_),
    inplace(sendbuf == recvbuf),
    host_sendbuf(inplace ?
                 nullptr :
                 get_pinned_memory<T>(send_displs_.back() + send_counts_.back())),
    host_recvbuf(get_pinned_memory<T>(recv_displs_.back() + recv_counts_.back())),
    send_counts(mpi::intify_size_t_vector(send_counts_)),
    send_displs(mpi::intify_size_t_vector(send_displs_)),
    recv_counts(mpi::intify_size_t_vector(recv_counts_)),
    recv_displs(mpi::intify_size_t_vector(recv_displs_)),
    comm(comm_.get_comm()) {
    // Transfer data from device to host.
    // We need to distinguish the inplace case in case the sendbuf is
    // larger than the recvbuf.
    // We may be able to optimize this into a single transfer in certain cases.
    if (inplace) {
      // If doing an in-place operation, transfer directly to host_recvbuf.
      for (size_t i = 0; i < recv_counts_.size(); ++i) {
        AL_CHECK_CUDA(cudaMemcpyAsync(host_recvbuf + recv_displs_[i],
                                      recvbuf + recv_displs_[i],
                                      sizeof(T)*recv_counts_[i],
                                      cudaMemcpyDeviceToHost, stream_));
      }
    } else {
      for (size_t i = 0; i < send_counts_.size(); ++i) {
        AL_CHECK_CUDA(cudaMemcpyAsync(host_sendbuf + send_displs_[i],
                                      sendbuf + send_displs_[i],
                                      sizeof(T)*send_counts_[i],
                                      cudaMemcpyDeviceToHost, stream_));
      }
    }
    start_event.record(stream_);

    // Have the device wait on the host.
    gpu_wait.wait(stream_);

    // Transfer completed buffer back to device.
    for (size_t i = 0; i < recv_counts_.size(); ++i) {
      AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf + recv_displs_[i],
                                    host_recvbuf + recv_displs_[i],
                                    sizeof(T)*recv_counts_[i],
                                    cudaMemcpyDeviceToHost, stream_));
    }
    end_event.record(stream_);
  }

  ~AlltoallvAlState() override {
    if (host_sendbuf) {
      release_pinned_memory(host_sendbuf);
    }
    release_pinned_memory(host_recvbuf);
  }

  std::string get_name() const override { return "HTAlltoallv"; }

protected:
  void start_mpi_op() override {
    MPI_Ialltoallv(inplace ? MPI_IN_PLACE : host_sendbuf,
                   send_counts.data(), send_displs.data(), mpi::TypeMap<T>(),
                   host_recvbuf,
                   recv_counts.data(), recv_displs.data(), mpi::TypeMap<T>(),
                   comm, get_mpi_req());
  }

private:
  bool inplace;
  T* host_sendbuf = nullptr;
  T* host_recvbuf = nullptr;
  std::vector<int> send_counts;
  std::vector<int> send_displs;
  std::vector<int> recv_counts;
  std::vector<int> recv_displs;
  MPI_Comm comm;
};

}  // namespace ht
}  // namespace internal
}  // namespace Al
