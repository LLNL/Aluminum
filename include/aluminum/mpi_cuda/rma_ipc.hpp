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

#include "aluminum/mpi_cuda/rma.hpp"
#include <cstring>
#include <set>

namespace Al {
namespace internal {
namespace mpi_cuda {

namespace rma_ipc {

class NotifyState: public AlState {
 public:
  NotifyState(AlRequest req, int peer, MPICUDACommunicator &comm,
              cudaEvent_t ev):
      AlState(req), m_peer(peer), m_comm(comm), m_ev(ev) {}
  void start() override {
    AlState::start();
    AL_CHECK_CUDA(cudaEventRecord(m_ev, m_comm.get_stream()));
    MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[1]);
  }
  PEAction step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    return flag ? PEAction::complete : PEAction::cont;
  }
 private:
  int key = 0;
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev;
  MPI_Request m_requests[2];
};

class WaitState: public AlState {
 public:
  WaitState(AlRequest req, int peer, MPICUDACommunicator &comm,
            cudaEvent_t ev_peer):
      AlState(req), m_peer(peer), m_comm(comm),
      m_ev_peer(ev_peer),
      m_stream_wait_set(false) {}
  void start() override {
    AlState::start();
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_req);
  }
  PEAction step() override {
    int flag;
    MPI_Test(&m_req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      if (!m_stream_wait_set) {
        AL_CHECK_CUDA(cudaStreamWaitEvent(
            m_comm.get_stream(), m_ev_peer, 0));
        MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
                  m_comm.get_comm(), &m_req);
        m_stream_wait_set = true;
        return PEAction::cont;
      } else {
        return PEAction::complete;
      }
    }
    return PEAction::cont;
  }
 private:
  int key = 0;
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev_peer;
  bool m_stream_wait_set;
  MPI_Request m_req;
};

class SyncState: public AlState {
 public:
  SyncState(AlRequest req, int peer, MPICUDACommunicator &comm,
            cudaEvent_t ev_self, cudaEvent_t ev_peer):
      AlState(req), m_peer(peer), m_comm(comm),
      m_ev_self(ev_self), m_ev_peer(ev_peer),
      m_stream_wait_set(false) {}
  void start() override {
    AlState::start();
    AL_CHECK_CUDA(cudaEventRecord(m_ev_self, m_comm.get_stream()));
    MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[0]);
    MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
              m_comm.get_comm(), &m_requests[1]);
  }
  PEAction step() override {
    int flag;
    MPI_Testall(2, m_requests, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      if (!m_stream_wait_set) {
        AL_CHECK_CUDA(cudaStreamWaitEvent(
            m_comm.get_stream(), m_ev_peer, 0));
        MPI_Isend(&key, 1, MPI_INT, m_peer, 0,
                  m_comm.get_comm(), &m_requests[0]);
        MPI_Irecv(&key, 1, MPI_INT, m_peer, 0,
                  m_comm.get_comm(), &m_requests[1]);
        m_stream_wait_set = true;
      } else {
        return PEAction::complete;
      }
    }
    return PEAction::cont;
  }
 private:
  int key = 0;
  int m_peer;
  MPICUDACommunicator &m_comm;
  cudaEvent_t m_ev_self;
  cudaEvent_t m_ev_peer;
  bool m_stream_wait_set;
  MPI_Request m_requests[2];
};

}

class ConnectionIPC: public Connection {
 public:
  ConnectionIPC(MPICUDACommunicator &comm, int peer, int dev):
      Connection(comm, peer), m_dev_peer(dev),
      m_peer_access_enabled(false), m_connected(false) {
    AL_CHECK_CUDA(cudaGetDevice(&m_dev));
    try_enable_peer_access();
    AL_CHECK_CUDA(cudaEventCreateWithFlags(
        &m_ev, cudaEventInterprocess | cudaEventDisableTiming));
  }

  ~ConnectionIPC() override {
    disconnect();
    AL_CHECK_CUDA(cudaEventDestroy(m_ev));
  }

  void connect() override {
    if (m_connected) return;
    cudaIpcEventHandle_t ipc_handle_self;
    cudaIpcEventHandle_t ipc_handle_peer;
    AL_CHECK_CUDA(cudaIpcGetEventHandle(&ipc_handle_self, m_ev));
    MPI_Sendrecv(&ipc_handle_self, sizeof(cudaIpcEventHandle_t),
                 MPI_BYTE, m_peer, 0,
                 &ipc_handle_peer, sizeof(cudaIpcEventHandle_t),
                 MPI_BYTE, m_peer, 0,
                 m_comm.get_comm(), MPI_STATUS_IGNORE);
    AL_CHECK_CUDA(
        cudaIpcOpenEventHandle(&m_ev_peer, ipc_handle_peer));
    m_connected = true;
  }

  void disconnect() override {
    if (!m_connected) return;
    detach_all_remote_buffers();
    AL_CHECK_CUDA(cudaEventDestroy(m_ev_peer));
    m_connected = false;
  }

  bool is_connected() const override {
    return m_connected;
  }

  void *attach_remote_buffer(void *local_addr) override {
    if (!m_connected) {
      connect();
    }
    cudaIpcMemHandle_t local_handle;
    cudaIpcMemHandle_t remote_handle;
    if (local_addr != nullptr) {
      AL_CHECK_CUDA(cudaIpcGetMemHandle(&local_handle, local_addr));
    } else {
      // Clears the handle if the local pointer is null
      std::memset(&local_handle, 0, sizeof(cudaIpcMemHandle_t));
    }
    MPI_Sendrecv(&local_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, m_peer, 0,
                 &remote_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, m_peer, 0,
                 get_mpi_comm(), MPI_STATUS_IGNORE);

    cudaIpcMemHandle_t void_handle;
    std::memset(&void_handle, 0, sizeof(cudaIpcMemHandle_t));
    // Remote address is not a valid address
    if (std::memcmp(&remote_handle, &void_handle,
                    sizeof(cudaIpcMemHandle_t)) == 0) {
      return nullptr;
    }
    if (!m_peer_access_enabled) {
      AL_CHECK_CUDA(cudaSetDevice(m_dev_peer));
    }
    void *mapped_buffer = nullptr;
    AL_CHECK_CUDA(
        cudaIpcOpenMemHandle(&mapped_buffer, remote_handle,
                             cudaIpcMemLazyEnablePeerAccess));
    if (!m_peer_access_enabled) {
      AL_CHECK_CUDA(cudaSetDevice(m_dev));
    }
    m_remote_buffers.insert(mapped_buffer);
    return mapped_buffer;
  }

  void detach_remote_buffer(void *remote_addr) override {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    if (remote_addr == nullptr) {
      return;
    }
    auto it = m_remote_buffers.find(remote_addr);
    if (it == m_remote_buffers.end()) {
      throw_al_exception("Invalid address");
    }
    if (!m_peer_access_enabled) {
      AL_CHECK_CUDA(cudaSetDevice(m_dev_peer));
    }
    AL_CHECK_CUDA(cudaIpcCloseMemHandle(remote_addr));
    if (!m_peer_access_enabled) {
      AL_CHECK_CUDA(cudaSetDevice(m_dev));
    }
    m_remote_buffers.erase(it);
  }

  void detach_all_remote_buffers() override {
    for (auto &x: m_remote_buffers) {
      detach_remote_buffer(x);
    }
  }

  void notify(AlRequest &req) {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    rma_ipc::NotifyState* state =
        new rma_ipc::NotifyState(req, m_peer, m_comm, m_ev);
    internal::get_progress_engine()->enqueue(state);
  }

  void wait(AlRequest &req) {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    rma_ipc::WaitState* state =
        new rma_ipc::WaitState(req, m_peer, m_comm, m_ev_peer);
    internal::get_progress_engine()->enqueue(state);
  }

  void sync(AlRequest &req) {
    if (!m_connected) {
      std::stringstream msg;
      msg << "Not connected to " << m_peer;
      throw_al_exception(msg.str());
    }
    rma_ipc::SyncState* state =
        new rma_ipc::SyncState(req, m_peer, m_comm, m_ev, m_ev_peer);
    internal::get_progress_engine()->enqueue(state);
  }

  void put(const void *src, void *dst, size_t size) override {
    if (!m_connected) {
      throw_al_exception("Not connected");
    }
    if (size > 0) {
      if (src == nullptr) {
        throw_al_exception("Source buffer is null");
      }
      if (dst == nullptr) {
        throw_al_exception("Destination buffer is null");
      }
      AL_CHECK_CUDA(cudaMemcpyPeerAsync(dst, m_dev_peer, src, m_dev,
                                        size, m_comm.get_stream()));
    }
  }

 private:
  int m_dev;
  int m_dev_peer;
  bool m_peer_access_enabled;
  bool m_connected;
  cudaEvent_t m_ev;
  cudaEvent_t m_ev_peer;
  std::set<void *> m_remote_buffers;

  void try_enable_peer_access() {
    if (!m_peer_access_enabled) {
      int peer_access;
      AL_CHECK_CUDA(
          cudaDeviceCanAccessPeer(&peer_access, m_dev, m_dev_peer));
      cudaError_t e = cudaDeviceEnablePeerAccess(m_dev_peer, 0);
      if (!(e == cudaSuccess ||
            e == cudaErrorPeerAccessAlreadyEnabled)) {
        throw_al_exception("Enabling peer access failed");
      }
      m_peer_access_enabled = true;
      // clear the error status
      cudaGetLastError();
    }
  }

};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
