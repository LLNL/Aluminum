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

#include "mpi_cuda/util.hpp"
#include "mpi_cuda/cuda_kernels.hpp"
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#define COLL_TOPOLOGY_OPT

namespace Al {
namespace internal {
namespace mpi_cuda {

/**
   Ring-based allreduce of GPU buffers.

   Requires CUDA-aware MPI. It seems that intra-node data transfer is
   more efficient with the CUDA runtime API even between devices
   managed by different MPI processes. This version uses the CUDA
   IPC-based memory transfer between intra-node devices. For
   inter-node transfer, MPI is used.

   In addition, bi-directional transfer can be enabled with the
   constructor.
 */
class RingMPICUDA {
 protected:
  enum TransferDir {L2R=0, R2L=1};
  const TransferDir DIRECTIONS[2] = {L2R, R2L};
  enum AccessType {MPI, PEER, HOST};
  enum Side {PREV=0, NEXT=1};
  const Side SIDES[2] = {PREV, NEXT};
  
  template <typename T>
  T get_opposite(T x) {
    return static_cast<T>(static_cast<int>(x) ^ 1);
  }
  
 public:
  RingMPICUDA(MPI_Comm comm): m_comm(comm) {
    COLL_CHECK_CUDA(cudaGetDevice(&m_gpu));
    build_ring();
    setup_events();
    COLL_CHECK_CUDA(cudaStreamCreate(&m_stream_r2l));
  }
  
  ~RingMPICUDA() {
    destroy_events();
    close_remote_buffer_mapping();
    free_gpu_bufs();
  }

 protected:


  // Rank reordering 
  void get_ring_indices() {
    const int local_size = get_mpi_comm_local_size();
#if defined(COLL_TOPOLOGY_OPT)
    if (local_size == 2 && m_np % 4 == 0) {
      get_ring_indices_topo_lp2();
      return;
    } else if (local_size == 4 && m_np % 8 == 0) {
      get_ring_indices_topo_lp4();
      return;
    } else {
      MPIPrintStream(std::cerr, m_pid)()
          << "Topology optimization is not supported for this number of processes per node: "
          << local_size << std::endl;
      // fall through to the default case below
    }
#else // default
    rid(L2R) = m_pid;
    prev_pid(L2R) = dec(m_pid, m_np);
    next_pid(L2R) = inc(m_pid, m_np);
    // R2L: 1 0 3 2
    if (local_size == 4) {
#ifdef AL_MPI_CUDA_MPI_LOAD_BALANCE
      const int id_offset_map[] = {1, -1, 1, -1};
      const int next_offset_map[] = {1, -3, 1, -3};
      const int prev_offset_map[] = {3, -1, 3, -1};
#else
      const int id_offset_map[] = {0, 0, 0, 0};
      const int next_offset_map[] = {-1, -1, -1, -1};
      const int prev_offset_map[] = {1, 1, 1, 1};
#endif
      int idx = m_pid % 4;
      rid(R2L) = m_pid + id_offset_map[idx];
      next_pid(R2L) =
          (m_pid + next_offset_map[idx] + m_np) % m_np;
      prev_pid(R2L) = (m_pid + prev_offset_map[idx]) % m_np;
    } else {
      // use the reverse ordering as L2R in the default case
      rid(R2L) = rid(L2R);
      next_pid(R2L) = prev_pid(L2R);
      prev_pid(R2L) = next_pid(L2R);
    }
#endif    
  }

  void get_ring_indices_topo_lp2() {
    // 0-1-3-2
#ifdef AL_DEBUG
    if (m_pid == 0) {
      MPIPrintStream(std::cout, m_pid)()
          << "Ring mapping for 2 ranks per node: "
          << "0-1-3-2" << std::endl;
    }
#endif        
    // this mapping assumes np is a multiple of 4.
    if ((m_np % 4) != 0) {
      MPIPrintStream(std::cout, m_pid)()
          << "Topology optimization requires process counts to be a multiple of 4: "
          << "#procs: " << m_np << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int id_offset_map[] = {0, 0, 1, -1};
    const int next_offset_map[] = {1, 2, 2, -1};
    const int prev_offset_map[] = {-2, -1, 1, -2};
    int idx = m_pid % 4;
    rid(L2R) = m_pid + id_offset_map[idx];
    next_pid(L2R) = (m_pid + next_offset_map[idx]) % m_np;
    prev_pid(L2R) =
        (m_pid + prev_offset_map[idx] + m_np) % m_np;
    rid(R2L) = rid(L2R);
    next_pid(R2L) = prev_pid(L2R);
    prev_pid(R2L) = next_pid(L2R);
  }

  void get_ring_indices_topo_lp4() {
    // L2R: 0-1-2-3-6-7-4-5
    // R2L: 1-0-3-2-7-6-5-4
#ifdef AL_DEBUG      
    if (m_pid == 0) {
      MPIPrintStream(std::cout, m_pid)()
          << "Ring mapping for 4 ranks per node: "
          << "0-1-2-3-6-7-4-5" << std::endl;
    }
#endif
    if ((m_np % 8) != 0) {
      MPIPrintStream(std::cout, m_pid)()
          << "Topology optimization requires process counts to be a multiple of 8: "
          << "#procs: " << m_np << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int idx = m_pid % 8;
    {
      // L2R
      TransferDir dir = L2R;
      const int id_offset_map[] = {0, 0, 0, 0, 2, 2, -2, -2};
      const int next_offset_map[] = {1, 1, 1, 3, 1, 3, 1, -3};
      const int prev_offset_map[] = {-3, -1, -1, -1, 3, -1, -3, -1};
      rid(dir) = m_pid + id_offset_map[idx];
      next_pid(dir) = (m_pid + next_offset_map[idx]) % m_np;
      prev_pid(dir) =
          (m_pid + prev_offset_map[idx] + m_np) % m_np;
    }
    {
      // R2L
      TransferDir dir = R2L;
#ifdef AL_MPI_CUDA_MPI_LOAD_BALANCE
      const int id_offset_map[] = {1, -1, 1, -1, 3, 1, -1, -3};
      const int next_offset_map[] = {1, -5, 1, -3, 1, 1, 1, -5};
      const int prev_offset_map[] = {3, -1, 5, -1, 5, -1, -1, -1};
#else
      const int id_offset_map[] = {0, 0, 0, 0, 2, 2, -2, -2};
      const int next_offset_map[] = {-3, -1, -1, -1, 3, -1, -3, -1};
      const int prev_offset_map[] = {1, 1, 1, 3, 1, 3, 1, -3};
#endif
      rid(dir) = m_pid + id_offset_map[idx];
      next_pid(dir) =
          (m_pid + next_offset_map[idx] + m_np) % m_np;
      prev_pid(dir) = (m_pid + prev_offset_map[idx]) % m_np;
    }
  }

  void setup_events() {
    for (TransferDir dir: DIRECTIONS) {
      cudaEvent_t ev;
      unsigned flag = cudaEventDisableTiming;
      if (prev_access_type(dir) != MPI) {
        flag |= cudaEventInterprocess;
      }
      COLL_CHECK_CUDA(cudaEventCreateWithFlags(&ev, flag));
      ready_ev(dir) = ev;
      flag = cudaEventDisableTiming;
      if (next_access_type(dir) != MPI) {
        flag |= cudaEventInterprocess;
      }
      COLL_CHECK_CUDA(cudaEventCreateWithFlags(&ev, flag));
      trans_ev(dir) = ev;
    }
    setup_inter_process_events();
    COLL_CHECK_CUDA(cudaEventCreateWithFlags(
        &r2l_ev(), cudaEventDisableTiming));
  }

  void destroy_events() {
    destroy_inter_process_events();
    // make sure exported events are first destroyed at remote
    // processes, though it may not be necessary.
    COLL_CHECK_MPI(MPI_Barrier(m_comm));
    for (TransferDir dir: DIRECTIONS) {
      COLL_CHECK_CUDA(cudaEventDestroy(ready_ev(dir)));
      COLL_CHECK_CUDA(cudaEventDestroy(trans_ev(dir)));
    }
    COLL_CHECK_CUDA(cudaEventDestroy(r2l_ev()));
  }

  void setup_inter_process_events() {
    // exchange inter-process events
    MPI_Request req[8];
    int req_idx = 0;
    cudaIpcEventHandle_t ipc_handles_peer[2][2];
    for (TransferDir dir: DIRECTIONS) {
      for (Side side: SIDES) {
        if (access_type(dir, side) != MPI) {
          int peer_id = neighbor_pid(dir, side);
          COLL_CHECK_MPI(MPI_Irecv(&ipc_handles_peer[dir][side],
                                   sizeof(cudaIpcEventHandle_t),
                                   MPI_BYTE, peer_id, 0, m_comm,
                                   &req[req_idx++]));
          cudaIpcEventHandle_t local_event_h;
          cudaEvent_t local_event = side == NEXT ?
              trans_ev(dir) : ready_ev(dir);
          COLL_CHECK_CUDA(cudaIpcGetEventHandle(&local_event_h,
                                                local_event));
          COLL_CHECK_MPI(MPI_Isend(&local_event_h,
                                   sizeof(cudaIpcEventHandle_t),
                                   MPI_BYTE, peer_id, 0, m_comm,
                                   &req[req_idx++]));
        }
      }
    }
    COLL_CHECK_MPI(MPI_Waitall(req_idx, req, MPI_STATUS_IGNORE));
    for (TransferDir dir: DIRECTIONS) {
      for (Side side: SIDES) {
        if (access_type(dir, side) != MPI) {
          COLL_CHECK_CUDA(cudaIpcOpenEventHandle(
              &neighbor_ev(dir, side), ipc_handles_peer[dir][side]));
        }
      }
    }
  }

  void destroy_inter_process_events() {
    for (TransferDir dir: DIRECTIONS) {
      for (Side side: SIDES) {
        if (access_type(dir, side) != MPI) {
          COLL_CHECK_CUDA(cudaEventDestroy(neighbor_ev(dir, side)));
        }
      }
    }
  }

  void build_ring() {
    COLL_CHECK_MPI(MPI_Comm_size(m_comm, &m_np));
    COLL_CHECK_MPI(MPI_Comm_rank(m_comm, &m_pid));  
    get_ring_indices();
    for (TransferDir dir: DIRECTIONS) {
      m_send_idx[dir] = rid(dir);
      int recv_idx = dir == L2R ?
          dec(m_send_idx[dir], m_np) :
          inc(m_send_idx[dir], m_np);
      m_recv_idx[dir] = recv_idx;
      // exchange device id with neighbor procs
      COLL_CHECK_MPI(MPI_Sendrecv(
          &m_gpu, 1, MPI_INT, next_pid(dir), 0,
          &prev_dev(dir), 1, MPI_INT, prev_pid(dir), 0,
          m_comm, MPI_STATUS_IGNORE));
      COLL_CHECK_MPI(MPI_Sendrecv(
          &m_gpu, 1, MPI_INT, prev_pid(dir), 0,
          &next_dev(dir), 1, MPI_INT, next_pid(dir), 0,
          m_comm, MPI_STATUS_IGNORE));
      setup_access_type(dir);
    }
  }

  void setup_access_type(TransferDir dir) {
    // Check whether RHS is in the same node.
    // If yes, the device ID of RHS should be the next one of this
    // process.
    // NOTE: This should be only valid if ranks are ordered by nodes

    // exchange node names
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    COLL_CHECK_MPI(MPI_Get_processor_name(proc_name, &name_len));
    char proc_name_peer[2][MPI_MAX_PROCESSOR_NAME];
    for (Side side: SIDES) {
      COLL_CHECK_MPI(MPI_Sendrecv(
          proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
          neighbor_pid(dir, get_opposite(side)), 0,
          proc_name_peer[side], MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
          neighbor_pid(dir, side), 0,
          m_comm, MPI_STATUS_IGNORE));
    }

    // Check whether the neighbor devices can be accessed with CUDA
    // API.
    for (Side side: SIDES) {
      if (std::strcmp(proc_name_peer[side], proc_name) != 0) {
        // Use MPI for communication between different nodes
        access_type(dir, side) = MPI;
        continue;
      }
      // Check peer access is possible
      int peer_access = 0;
      int peer_dev = neighbor_dev(dir, side);
      COLL_CHECK_CUDA(cudaDeviceCanAccessPeer(
          &peer_access, m_gpu, peer_dev));
      if (!peer_access) {
        access_type(dir, side) = HOST;
        continue;
      }
#ifdef AL_DEBUG
      MPIPrintStream(std::cerr, m_pid)()
          << "Enabling peer access; local_dev: "
          << m_gpu << ", peer dev: " << peer_dev << "\n";
#endif          
      cudaError_t err = cudaDeviceEnablePeerAccess(peer_dev, 0);
      if (err != cudaSuccess &&
          err != cudaErrorPeerAccessAlreadyEnabled) {
        // This case is known to happen when the compute mode
        // disallows to share devices from multiple processes.
        // Mask error status so that it doesn't show up again.
        cudaGetLastError();  
        // Fall back to host communication.
        access_type(dir, side) = HOST;
        continue;
      }
      // Use peer access
      access_type(dir, side) = PEER;
    }
  }


  template <typename T>
  void get_gpu_bufs(size_t count, T **bufs) {
    size_t real_size = sizeof(T) * count;
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;
      if (!(m_gpu_bufs[dir] != nullptr &&
            m_gpu_buf_sizes[dir] >= real_size)) {
#ifdef AL_DEBUG
        MPIPrintStream(std::cerr, m_pid)()
            << "Setting up a new workspace buffer for "
            << (dir == L2R ? "L2R" : "R2L")
            << ". current size: " << m_gpu_buf_sizes[dir]
            << ", new size: " << real_size << "\n";
#endif
        close_remote_buffer_mapping(dir);
        // Make sure everyone unmaps the IPC handle before actually
        // freeing the memory
        COLL_CHECK_MPI(MPI_Barrier(m_comm));
        free_gpu_buf(dir);
        void *p = nullptr;
        COLL_CHECK_CUDA(cudaMalloc(&p, real_size));
        COLL_ASSERT(p != nullptr);
        m_gpu_bufs[dir] = p;
        m_gpu_buf_sizes[dir] = real_size;
        setup_remote_buffer_mapping(dir);
      }
      bufs[dir] = static_cast<T*>(m_gpu_bufs[dir]);
    }
  }

  void free_gpu_buf(TransferDir dir) {
    if (m_gpu_bufs[dir] != nullptr) {
      COLL_CHECK_CUDA(cudaFree(m_gpu_bufs[dir]));
      m_gpu_bufs[dir] = nullptr;
    }
  }

  void free_gpu_bufs() {
    for (TransferDir dir: DIRECTIONS) {
      free_gpu_buf(dir);
    }
  }
  
  void setup_remote_buffer_mapping(TransferDir dir) {
    if (!m_trans_dir[dir]) return;
    cudaIpcMemHandle_t local_ipc_h, peer_ipc_h;
    // If the peer can access the local device with CUDA API,
    // expose its buffer address as an IPC handle
    size_t send_msg_size = 0;
    if (prev_access_type(dir) != MPI) {
      COLL_CHECK_CUDA(cudaIpcGetMemHandle(
          &local_ipc_h, m_gpu_bufs[dir]));
      send_msg_size = sizeof(cudaIpcMemHandle_t);
    }
    size_t recv_msg_size =
        next_access_type(dir) != MPI ? sizeof(cudaIpcMemHandle_t) : 0;
    COLL_CHECK_MPI(MPI_Sendrecv(
        &local_ipc_h, send_msg_size,  MPI_BYTE, prev_pid(dir), 0,
        &peer_ipc_h, recv_msg_size, MPI_BYTE, next_pid(dir), 0,
        m_comm, MPI_STATUS_IGNORE));
    if (next_access_type(dir) != MPI) {
      change_cur_device_to_accessible_to_next_dev(dir);
      COLL_CHECK_CUDA(cudaIpcOpenMemHandle(
          &next_work(dir), peer_ipc_h,
          cudaIpcMemLazyEnablePeerAccess));
      reset_cur_device();
    }
  }

  void close_remote_buffer_mapping(TransferDir dir) {
    if (next_work(dir) != nullptr) {
      change_cur_device_to_accessible_to_next_dev(dir);
      COLL_CHECK_CUDA(cudaIpcCloseMemHandle(next_work(dir)));
      next_work(dir) = nullptr;
      reset_cur_device();
    }
  }

  void close_remote_buffer_mapping() {
    for (TransferDir dir: DIRECTIONS) {
      close_remote_buffer_mapping(dir);
    }
  }

  size_t setup_pe_counts(size_t count, std::vector<size_t> *pe_counts,
                         std::vector<size_t> *pe_offsets) {
    int num_directions = 0;
    for (TransferDir dir: DIRECTIONS) {
      if (m_trans_dir[dir]) ++num_directions;
    }
    size_t pe_count_base = count / (m_np * num_directions);
    int rem = count % (m_np * num_directions);
    int idx_offset = 0;
    size_t cur_pe_offset = 0;
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;
      for (int j = 0; j < m_np; ++j) {
        int idx = j + idx_offset;
        pe_counts[dir].push_back(pe_count_base + ((idx < rem) ? 1 : 0));
        pe_offsets[dir].push_back(cur_pe_offset);
        cur_pe_offset += pe_counts[dir][j];
      }
      idx_offset += m_np;
    }
    size_t max_count = pe_count_base + (rem > 0 ? 1 : 0);
    return max_count;
  }

  void notify(int rank, int tag) {
    char x = 0;
    COLL_CHECK_MPI(MPI_Send(
        &x, 1, MPI_CHAR, rank, tag, m_comm));
  }

  void notify_next_proc(TransferDir dir) {
    notify(next_pid(dir), notification_next_tag(dir));
  }
  
  void notify_prev_proc(TransferDir dir) {
    notify(prev_pid(dir), notification_prev_tag(dir));
  }

  void wait_for_notification(int rank, int tag) {
    char x;
    COLL_CHECK_MPI(MPI_Recv(
        &x, 1, MPI_CHAR, rank, tag,
        m_comm, MPI_STATUS_IGNORE));
  }

  void wait_for_prev_proc(TransferDir dir) {
    wait_for_notification(prev_pid(dir), notification_next_tag(dir));
  }

  void wait_for_next_proc(TransferDir dir) {
    wait_for_notification(next_pid(dir), notification_prev_tag(dir));
  }

  template <typename T>
  void transfer(T *buf, T *work_buf,
                size_t send_count, size_t recv_count,
                size_t send_offset,
                cudaStream_t stream, int iter_idx,
                MPI_Request *send_requests, int &num_send_requests,
                MPI_Request *recv_requests, int &num_recv_requests,
                const TransferDir dir) {
    const int tag = 0;
    const MPI_Datatype mpi_type = get_mpi_data_type<T>();
    num_send_requests = 0;
    num_recv_requests = 0;
    if (prev_access_type(dir) == MPI) {
      COLL_CHECK_CUDA(cudaEventSynchronize(ready_ev(dir)));
      COLL_CHECK_MPI(MPI_Irecv(
          work_buf, recv_count, mpi_type, prev_pid(dir), tag,
          m_comm, &recv_requests[num_recv_requests++]));
    }
    // Send to neighbor        
    T *src_ptr = buf + send_offset;
    if (next_access_type(dir) == MPI) {
      // already synchronized if prev_access_type == MPI
      if (prev_access_type(dir) != MPI) {
        COLL_CHECK_CUDA(cudaEventSynchronize(ready_ev(dir)));
      }
      COLL_CHECK_MPI(MPI_Isend(
          src_ptr, send_count, mpi_type,
          next_pid(dir), tag, m_comm,
          &send_requests[num_send_requests++]));
    } else {
      // Make sure the completion event of computation was
      // recorded before sending the buffer
      if (iter_idx != 0) {
        wait_for_next_proc(dir);
      }
      COLL_CHECK_CUDA(cudaStreamWaitEvent(
          stream, next_ev(dir), 0));
      COLL_CHECK_CUDA(cudaMemcpyPeerAsync(
          next_work(dir), next_dev(dir), src_ptr, m_gpu,
          send_count * sizeof(T), stream));
      COLL_CHECK_CUDA(cudaEventRecord(trans_ev(dir), stream));
      notify_next_proc(dir);
    }
  }

  void ensure_recv(MPI_Request *recv_requests, int num_requests,
                   cudaStream_t *streams) {
    // Set dependency for transfer from the neighbor device
    for (TransferDir dir: DIRECTIONS) {
      if (m_trans_dir[dir] && prev_access_type(dir) != MPI) {
        wait_for_prev_proc(dir);
        COLL_CHECK_CUDA(cudaStreamWaitEvent(
            streams[dir], prev_ev(dir), 0));
      }
    }
    // Wait for completion of MPI transfer
    COLL_CHECK_MPI(MPI_Waitall(
        num_requests, recv_requests, MPI_STATUS_IGNORE));
  }

  template <typename T>
  void issue_local_reduction(int step, T *buf, T *work, size_t count,
                             cudaStream_t stream, ReductionOperator op,
                             TransferDir dir) {
    if (step == 0) {
      reduce1(buf, work, count, stream, op,
              GetReductionOperandType<T>::key);
    } else {
      COLL_CHECK_CUDA(cudaMemcpyAsync(
          buf, work, count * sizeof(T), cudaMemcpyDeviceToDevice,
          stream));
    }
    COLL_CHECK_CUDA(cudaEventRecord(ready_ev(dir), stream));
    if (prev_access_type(dir) != MPI) notify_prev_proc(dir);
  }

  void ensure_send(MPI_Request *send_requests, int num_requests) {
    COLL_CHECK_MPI(MPI_Waitall(num_requests, send_requests,
                               MPI_STATUS_IGNORE));
  }

  void update_indices(int *send_idx, int *recv_idx) {
    send_idx[L2R] = dec(send_idx[L2R], m_np);
    recv_idx[L2R] = dec(recv_idx[L2R], m_np);
    send_idx[R2L] = inc(send_idx[R2L], m_np);
    recv_idx[R2L] = inc(recv_idx[R2L], m_np);
  }
  
 public:
  // buf: GPU buffers
  template <typename T>
  int allreduce(T *buf, size_t count, ReductionOperator op,
                cudaStream_t st, bool bidirectional=true) {
    if (count == 0) return 0;
    
    // Set whether the second direction is used
    m_trans_dir[R2L] = bidirectional;    

    std::vector<size_t> pe_counts[2];
    std::vector<size_t> pe_offsets[2];
    const size_t max_pe_count = setup_pe_counts(
        count, pe_counts, pe_offsets);
    T *work_bufs[2];
    get_gpu_bufs<T>(max_pe_count, work_bufs);

    int send_idx[2] = {m_send_idx[0], m_send_idx[1]};
    int recv_idx[2] = {m_recv_idx[0], m_recv_idx[1]};

    
    cudaStream_t streams[2] = {st, m_stream_r2l};
    if (bidirectional) {
      // Make sure the R2L stream does not go ahead until the pending
      // L2R operations are done.
      COLL_CHECK_CUDA(cudaEventRecord(
          r2l_ev(), streams[L2R]));
      COLL_CHECK_CUDA(cudaStreamWaitEvent(
          streams[R2L], r2l_ev(), 0));
    }

    // Make sure transfer with MPI waits for pending tasks on the
    // stream. Not necessary for PEER/HOST as it is synchronized with
    // the stream
    for (TransferDir dir: DIRECTIONS) {
      if (next_access_type(dir) == MPI) {
        COLL_CHECK_CUDA(cudaEventRecord(ready_ev(dir), streams[dir]));
      }
    }

    // Step 1: Reduce-scatter
    // Step 2: Allgather
    for (int step = 0; step < 2; ++step) {    
      for (int i = 0; i < m_np - 1; ++i) {
        MPI_Request send_requests[2];
        MPI_Request recv_requests[2];
        int num_send_requests = 0;
        int num_recv_requests = 0;
        // Issue transfer operations, which can be cudaMemcpy or
        // MPI_Isend
        for (TransferDir dir: DIRECTIONS) {
          if (!m_trans_dir[dir]) continue;          
          int nr_send, nr_recv;
          transfer(buf, work_bufs[dir],
                   pe_counts[dir][send_idx[dir]],
                   pe_counts[dir][recv_idx[dir]],
                   pe_offsets[dir][send_idx[dir]],
                   streams[dir], i,
                   send_requests + num_send_requests, nr_send,
                   recv_requests + num_recv_requests, nr_recv,
                   dir);
          num_send_requests += nr_send;
          num_recv_requests += nr_recv;
        }

        // First, issues local reductions that do not depend on
        // MPI. MPI-dependent reductions require MPI_Wait, which
        // blocks the host thread, they are taken care after issuing
        // non MPI-dependent operations
        for (TransferDir dir: DIRECTIONS) {
          if (!m_trans_dir[dir] ||
              prev_access_type(dir) == MPI) continue;
          wait_for_prev_proc(dir);
          COLL_CHECK_CUDA(cudaStreamWaitEvent(
              streams[dir], prev_ev(dir), 0));
          issue_local_reduction(
              step, buf + pe_offsets[dir][recv_idx[dir]],
              work_bufs[dir], pe_counts[dir][recv_idx[dir]],
              streams[dir], op, dir);
        }

        // Issues local reductions that use data sent with MPI
        COLL_CHECK_MPI(MPI_Waitall(
            num_recv_requests, recv_requests, MPI_STATUS_IGNORE));
        for (TransferDir dir: DIRECTIONS) {
          if (!m_trans_dir[dir] ||
              prev_access_type(dir) != MPI) continue;
          issue_local_reduction(
              step, buf + pe_offsets[dir][recv_idx[dir]],
              work_bufs[dir], pe_counts[dir][recv_idx[dir]],
              streams[dir], op, dir);
        }

        // Cleans up MPI_Isend requests
        ensure_send(send_requests, num_send_requests);
        
        update_indices(send_idx, recv_idx);
      }

      for (TransferDir dir: DIRECTIONS) {
        if (!m_trans_dir[dir]) continue;
        if (next_access_type(dir) != MPI) {
          wait_for_next_proc(dir);
        }
      }
    }

    // Wait for completion of R2L
    if (m_trans_dir[R2L]) {
      COLL_CHECK_CUDA(cudaEventRecord(
          r2l_ev(), streams[R2L]));
      COLL_CHECK_CUDA(cudaStreamWaitEvent(
          streams[L2R], r2l_ev(), 0));
    }

    return 0;
  }

 protected:

  int &rid(TransferDir dir) {
    return m_rid[dir];
  }

  int &neighbor_pid(TransferDir dir, Side s) {
    return m_neighbor_pids[dir][s];
  }
  
  int &prev_pid(TransferDir dir) {
    return neighbor_pid(dir, PREV);
  }

  int &next_pid(TransferDir dir) {
    return neighbor_pid(dir, NEXT);    
  }

  AccessType &access_type(TransferDir dir, Side s) {
    return m_access_types[dir][s];
  }

  AccessType &prev_access_type(TransferDir dir) {
    return access_type(dir, PREV);
  }

  AccessType &next_access_type(TransferDir dir) {
    return access_type(dir, NEXT);
  }

  int &neighbor_dev(TransferDir dir, Side s) {
    return m_neighbor_dev[dir][s];
  }

  int &next_dev(TransferDir dir) {
    return neighbor_dev(dir, NEXT);
  }

  int &prev_dev(TransferDir dir) {
    return neighbor_dev(dir, PREV);
  }

  void *&next_work(TransferDir dir) {
    return m_neighbor_work[dir];
  }

  cudaEvent_t &ready_ev(TransferDir dir) {
    return m_ev_ready[dir];
  }

  cudaEvent_t &r2l_ev() {
    return m_ev_r2l;
  }

  cudaEvent_t &trans_ev(TransferDir dir) {
    return m_ev_trans[dir];
  }

  cudaEvent_t &neighbor_ev(TransferDir dir, Side s) {
    return m_neighbor_ev[dir][s];
  }

  cudaEvent_t &next_ev(TransferDir dir) {
    return neighbor_ev(dir, NEXT);
  }

  cudaEvent_t &prev_ev(TransferDir dir) {
    return neighbor_ev(dir, PREV);
  }

  int notification_next_tag(TransferDir dir) {
    return m_notification_next_tags[dir];
  }

  int notification_prev_tag(TransferDir dir) {
    return m_notification_prev_tags[dir];
  }

  // current device setting
  void reset_cur_device() {
    COLL_CHECK_CUDA(cudaSetDevice(m_gpu));
  }

  void change_cur_device_to_accessible_to_next_dev(TransferDir dir) {
    // If it needs to be accessed through host memory, open the
    // IPC handle at a context on the remote GPU
    if (next_access_type(dir) == HOST) {
#ifdef AL_DEBUG
      MPIPrintStream(std::cerr, m_pid)()
          << "Opening a context on a remote GPU, "
          << next_dev(dir) << ", from process " << m_pid << "\n";
#endif
      COLL_CHECK_CUDA(cudaSetDevice(next_dev(dir)));
    }
  }

  MPI_Comm m_comm;  
  void *m_gpu_bufs[2] = {nullptr, nullptr};
  size_t m_gpu_buf_sizes[2] = {0, 0};
  cudaStream_t m_stream_r2l = 0;
  cudaEvent_t m_ev_r2l;

  int m_gpu;
  int m_np;
  int m_pid;
  int m_rid[2];
  int m_neighbor_pids[2][2];  
  int m_send_idx[2];
  int m_recv_idx[2];

  cudaEvent_t m_ev_ready[2];
  cudaEvent_t m_ev_trans[2];

  AccessType m_access_types[2][2];
  int m_neighbor_dev[2][2];
  void *m_neighbor_work[2] = {nullptr, nullptr};
  cudaEvent_t m_neighbor_ev[2][2];

  bool m_trans_dir[2] = {true, true};

  int m_notification_next_tags[2] = {1001, 1002};
  int m_notification_prev_tags[2] = {1003, 1004};
};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
