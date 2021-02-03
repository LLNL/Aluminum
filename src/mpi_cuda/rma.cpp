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

#include "Al.hpp"
#include "aluminum/mpi_cuda/rma.hpp"
#include "aluminum/mpi_cuda/rma_ipc.hpp"
#include "aluminum/mpi_cuda/rma_self.hpp"
#include "aluminum/mpi_cuda/rma_null.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

int RMA::get_local_peer_device(int peer) {
  int peer_local_rank = get_peer_local_rank(peer);
  if (peer_local_rank == -1) return -1;
  return m_local_devices[peer_local_rank];
}

bool RMA::is_ipc_capable(int peer) {
  if (!is_on_same_node(peer)) return false;

  int peer_dev = get_local_peer_device(peer);
  int peer_access;
  AL_CHECK_CUDA(
      cudaDeviceCanAccessPeer(&peer_access, m_dev, peer_dev));
  if (peer_access != 0) {
    // Peer access possible
    return true;
  }
  // IPC memcpy is still possible if a context can be created
  // at the remote device
  cudaDeviceProp prop;
  AL_CHECK_CUDA(
      cudaGetDeviceProperties(&prop, peer_dev));
  if (prop.computeMode == cudaComputeModeDefault) {
    return true;
  }
  return false;
}

void RMA::open_connection(int peer) {
  if (find_connection(peer)) return;
  Connection *new_conn = nullptr;
  if (peer == MPI_PROC_NULL) {
    new_conn = new ConnectionNULL(m_comm, peer);
  } else if (m_comm.rank() == peer) {
    new_conn = new ConnectionSelf(m_comm, peer);
  } else if (is_ipc_capable(peer)) {
    new_conn = new ConnectionIPC(m_comm, peer, get_local_peer_device(peer));
  } else {
    throw_al_exception("Cannot connect");
  }
  m_connections.insert(std::make_pair(peer, new_conn));
}

Connection *RMA::find_connection(int peer) {
  auto conn = m_connections.find(peer);
  if (conn == m_connections.end()) {
    return nullptr;
  }
  return conn->second;
}

Connection *RMA::get_connection(int peer) {
  open_connection(peer);
  auto conn = find_connection(peer);
  if (!conn) {
    throw_al_exception("Connection not available");
  }
  return conn;
}

void RMA::close_all_connections() {
  bool disconnect_called = false;
  for (auto p: m_connections) {
    if (p.second->is_connected()) {
      disconnect_called = true;
      p.second->disconnect();
    }
  }
  if (!disconnect_called) return;
  MPI_Barrier(m_comm.get_comm());
  for (auto p: m_connections) {
    delete p.second;
  }
  m_connections.clear();
}

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
