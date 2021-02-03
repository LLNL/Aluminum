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

#include "aluminum/mpi_cuda/communicator.hpp"
#include "aluminum/cuda.hpp"
#include <map>

namespace Al {
namespace internal {
namespace mpi_cuda {

class Connection {
 public:
  Connection(MPICUDACommunicator &comm, int peer):
      m_comm(comm), m_peer(peer) {}
  virtual ~Connection() {}
  virtual void connect() = 0;
  virtual void disconnect() {}
  virtual bool is_connected() const {
    return false;
  }
  virtual void *attach_remote_buffer(void *local_addr) = 0;
  virtual void detach_remote_buffer(void *remote_addr) = 0;
  virtual void detach_all_remote_buffers() = 0;
  virtual void notify(AlRequest &req) = 0;
  virtual void wait(AlRequest &req) = 0;
  virtual void sync(AlRequest &req) = 0;
  virtual void put(const void *src, void *dst,
                   size_t size) = 0;

 protected:
  MPICUDACommunicator &m_comm;
  int m_peer;

  MPI_Comm get_mpi_comm() {
    return m_comm.get_comm();
  }
};

class RMA {
 private:
  MPICUDACommunicator &m_comm;
  MPI_Group m_group;
  MPI_Group m_local_group;
  int m_dev;
  std::vector<int> m_local_devices;
  std::map<int, Connection *> m_connections;

 public:
  RMA(MPICUDACommunicator &comm): m_comm(comm) {
    MPI_Comm_group(m_comm.get_comm(), &m_group);
    MPI_Comm local_comm;
    MPI_Comm_split_type(m_comm.get_comm(), MPI_COMM_TYPE_SHARED,
                        0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_group(local_comm, &m_local_group);
    AL_CHECK_CUDA(cudaGetDevice(&m_dev));
    int num_local_ranks;
    MPI_Comm_size(local_comm, &num_local_ranks);
    m_local_devices.reserve(num_local_ranks);
    MPI_Allgather(&m_dev, 1, MPI_INT,
                  m_local_devices.data(), 1, MPI_INT,
                  m_comm.get_comm());
    MPI_Comm_free(&local_comm);
  }

  ~RMA() {
    MPI_Group_free(&m_group);
    MPI_Group_free(&m_local_group);
    close_all_connections();
  }

  int get_peer_local_rank(int peer) {
    int ranks[2] = {m_comm.rank(), peer};
    int translated_ranks[2];
    MPI_Group_translate_ranks(m_group, 2, ranks, m_local_group, translated_ranks);
    if (translated_ranks[1] == MPI_UNDEFINED || translated_ranks[1] == MPI_PROC_NULL) {
      return -1;
    } else {
      return translated_ranks[1];
    }
  }

  bool is_on_same_node(int peer) {
    return get_peer_local_rank(peer) != -1;
  }

  int get_local_peer_device(int peer);
  bool is_ipc_capable(int peer);
  void open_connection(int peer);
  Connection *find_connection(int peer);
  Connection *get_connection(int peer);
  void close_all_connections();

  void *attach_remote_buffer(void *local_addr, int peer) {
    auto conn = get_connection(peer);
    return conn->attach_remote_buffer(local_addr);
  }

  void detach_remote_buffer(void *remote_addr, int peer) {
    auto conn = get_connection(peer);
    conn->detach_remote_buffer(remote_addr);
  }

  void detach_all_remote_buffers(int peer) {
    auto conn = get_connection(peer);
    conn->detach_all_remote_buffers();
  }

  void put(const void *src, int dst_rank, void *dst,
           size_t size) {
    auto conn = get_connection(dst_rank);
    conn->put(src, dst, size);
  }

  void notify(int dst_rank) {
    auto conn = get_connection(dst_rank);
    AlRequest req = internal::get_free_request();
    conn->notify(req);
    internal::get_progress_engine()->wait_for_completion(req);
  }

  void wait(int dst_rank) {
    auto conn = get_connection(dst_rank);
    AlRequest req = internal::get_free_request();
    conn->wait(req);
    internal::get_progress_engine()->wait_for_completion(req);
  }

  void sync(int peer) {
    auto conn = get_connection(peer);
    AlRequest req = internal::get_free_request();
    conn->sync(req);
    internal::get_progress_engine()->wait_for_completion(req);
  }

  void sync(const int *peers, int num_peers) {
    AlRequest *requests = new AlRequest[num_peers];
    for (int i = 0; i < num_peers; ++i) {
      auto conn = get_connection(peers[i]);
      AlRequest req = internal::get_free_request();
      conn->sync(req);
      requests[i] = req;
    }
    for (int i = 0; i < num_peers; ++i) {
      internal::get_progress_engine()->wait_for_completion(requests[i]);
    }
  }

};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
