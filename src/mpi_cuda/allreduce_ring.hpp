#pragma once

#include "mpi_cuda/util.hpp"
#include "mpi_cuda/cuda_kernels.hpp"
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

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

   Tested with MVAPICH2-2.3b (without GDR) with one GPU per MPI rank,
   though should work with multiple GPUs per MPI rank if supported by
   MPI.
 */
class RingMPICUDA {
 protected:
  enum TransferDir {L2R=0, R2L=1};
  const TransferDir DIRECTIONS[2] = {L2R, R2L};
  enum AccessType {MPI, PEER, HOST};
  enum Side {LHS=0, RHS=1};
  const Side SIDES[2] = {LHS, RHS};
  
 public:
  RingMPICUDA(MPI_Comm comm): m_comm(comm) {
    COLL_CHECK_CUDA(cudaGetDevice(&m_gpu));
    build_ring();
    setup_events();
  }
  
  ~RingMPICUDA() {
    destroy_events();
    close_remote_buffer_mapping();
    free_gpu_bufs();
  }

 protected:


  // Rank reordering 
  void get_ring_indices(int self, int num_procs,
                        int &ring_id, int &ring_lhs,
                        int &ring_rhs) {
#if defined(COLL_TOPOLOGY_OPT_RAY)
    int local_size = get_mpi_comm_local_size();
    if (local_size == 2) {
      // 0-1-3-2
#ifdef AL_DEBUG
      if (self == 0) {      
        MPIPrintStream(std::cout, self)()
            << "Ring mapping for 2 ranks per node: "
            << "0-1-3-2" << std::endl;
      }
#endif        
      // this mapping assumes np is a multiple of 4.
      COLL_ASSERT((num_procs % 4) == 0);
      // avoids conflicts on the Ray IB topology
      const int id_offset_map[] = {0, 0, 1, -1};    
      const int rhs_offset_map[] = {1, 2, 2, -1};
      const int lhs_offset_map[] = {-2, -1, 1, -2};
      int idx = self % 4;
      ring_id = self + id_offset_map[idx];
      ring_rhs = self + rhs_offset_map[idx];
      ring_lhs = self + lhs_offset_map[idx];
      if (self == 0) ring_lhs = num_procs - 2;
      if (self == num_procs - 2) ring_rhs = 0;
    } else if (local_size == 4) {
#ifdef AL_DEBUG      
      // 0-1-2-3-6-7-4-5
      if (self == 0) {
        MPIPrintStream(std::cout, self)()
            << "Ring mapping for 4 ranks per node: "
            << "0-1-2-3-6-7-4-5" << std::endl;
      }
#endif
      COLL_ASSERT((num_procs % 8) == 0);
      const int id_offset_map[] = {0, 0, 0, 0, 2, 2, -2, -2};
      const int rhs_offset_map[] = {1, 1, 1, 3, 1, 3, 1, -3};
      const int lhs_offset_map[] = {-3, -1, -1, -1, 3, -1, -3, -1};
      int idx = self % 8;
      ring_id = self + id_offset_map[idx];
      ring_rhs = self + rhs_offset_map[idx];
      ring_lhs = self + lhs_offset_map[idx];
      if (self == 0) ring_lhs = num_procs - 3;
      if (self == num_procs - 3) ring_rhs = 0;      
    } else {
      MPIPrintStream(std::cerr, self)() << "Unsupported number of processes per rank: " << local_size << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
#else // default
    ring_id = self;
    ring_lhs = dec(self, num_procs);
    ring_rhs = inc(self, num_procs);
#endif    
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

  void setup_events() {
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;
      cudaEvent_t ev;
      unsigned flag = cudaEventDisableTiming;
      if (prev_access_type(dir) != MPI) {
        flag |= cudaEventInterprocess;
      }
      COLL_CHECK_CUDA(cudaEventCreateWithFlags(&ev, flag));
      comp_ev(dir) = ev;
      flag = cudaEventDisableTiming;
      if (next_access_type(dir) != MPI) {
        flag |= cudaEventInterprocess;
      }
      COLL_CHECK_CUDA(cudaEventCreateWithFlags(&ev, flag));
      trans_ev(dir) = ev;
    }
    setup_inter_process_events();
  }

  void destroy_events() {
    destroy_inter_process_events();
    // make sure exported events are first destroyed at remote
    // processes, though it may not be necessary.
    COLL_CHECK_MPI(MPI_Barrier(m_comm));
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;        
      COLL_CHECK_CUDA(cudaEventDestroy(comp_ev(dir)));
      COLL_CHECK_CUDA(cudaEventDestroy(trans_ev(dir)));
    }
  }

  void build_ring() {
    COLL_CHECK_MPI(MPI_Comm_size(m_comm, &m_np));
    COLL_CHECK_MPI(MPI_Comm_rank(m_comm, &m_pid));  
    get_ring_indices(m_pid, m_np, rid(), pid(LHS), pid(RHS));
    for (TransferDir dir: DIRECTIONS) {
      m_send_idx[dir] = m_rid;
      int recv_idx = dir == L2R ?
          dec(m_send_idx[dir], m_np) :
          inc(m_send_idx[dir], m_np);
      m_recv_idx[dir] = recv_idx;
    }
    // exchange device id with RHS
    COLL_CHECK_MPI(MPI_Sendrecv(
        &m_gpu, 1, MPI_INT, pid(RHS), 0,
        &neighbor_dev(LHS), 1, MPI_INT, pid(LHS), 0,
        m_comm, MPI_STATUS_IGNORE));
    COLL_CHECK_MPI(MPI_Sendrecv(
        &m_gpu, 1, MPI_INT, pid(LHS), 0,
        &neighbor_dev(RHS), 1, MPI_INT, pid(RHS), 0,
        m_comm, MPI_STATUS_IGNORE));
    setup_access_type();
  }
  
  void setup_access_type() {
    // Check whether RHS is in the same node.
    // If yes, the device ID of RHS should be the next one of this
    // process.
    // NOTE: This should be only valid if ranks are ordered by nodes

    // exchange node names
    char proc_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    COLL_CHECK_MPI(MPI_Get_processor_name(proc_name, &name_len));
    char proc_name_lhs[MPI_MAX_PROCESSOR_NAME];
    char proc_name_rhs[MPI_MAX_PROCESSOR_NAME];    
    COLL_CHECK_MPI(MPI_Sendrecv(
        proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, pid(RHS), 0,
        proc_name_lhs, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, pid(LHS), 0,
        m_comm, MPI_STATUS_IGNORE));
    COLL_CHECK_MPI(MPI_Sendrecv(
        proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, pid(LHS), 0,
        proc_name_rhs, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, pid(RHS), 0,
        m_comm, MPI_STATUS_IGNORE));

    // Check whether the neighbor devices can be accessed with CUDA
    // API.
    for (Side side: SIDES) {
      char *peer_proc_name = side == LHS ? proc_name_lhs : proc_name_rhs;
      if (std::strcmp(peer_proc_name, proc_name) == 0) {
        int peer_access = 0;
        COLL_CHECK_CUDA(cudaDeviceCanAccessPeer(
            &peer_access, m_gpu, neighbor_dev(side)));
        if (peer_access) {
#ifdef AL_DEBUG
          MPIPrintStream(std::cerr, m_pid)()
              << "enabling peer access; local_dev: "
              << m_gpu << ", peer dev: " << neighbor_dev(side) << "\n";
#endif          
          cudaError_t err = cudaDeviceEnablePeerAccess(neighbor_dev(side), 0);
          if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();  // So we don't catch this error later.
            // Fall back to host communication.
            access_type(side) = HOST;
          } else {
            access_type(side) = PEER;
          }
        } else {
          access_type(side) = HOST;        
        }
      } else {
        access_type(side) = MPI;
      }
    }
  }

  void setup_inter_process_events() {
    // exchange inter-process events
    MPI_Request req[8];
    int req_idx = 0;
    cudaIpcEventHandle_t ipc_handles_next[2];
    cudaIpcEventHandle_t ipc_handles_prev[2];    
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;
      if (next_access_type(dir) != MPI) {
        int peer_id = next_pid(dir);
        COLL_CHECK_MPI(MPI_Irecv(&ipc_handles_next[dir],
                                 sizeof(cudaIpcEventHandle_t), MPI_BYTE,
                                 peer_id, 0, m_comm, &req[req_idx++]));
        cudaIpcEventHandle_t local_event_h;
        COLL_CHECK_CUDA(cudaIpcGetEventHandle(&local_event_h,
                                              trans_ev(dir)));
        COLL_CHECK_MPI(MPI_Isend(&local_event_h,
                                 sizeof(cudaIpcEventHandle_t), MPI_BYTE,
                                 peer_id, 0, m_comm, &req[req_idx++]));
      }
      if (prev_access_type(dir) != MPI) {
        int peer_id = prev_pid(dir);
        COLL_CHECK_MPI(MPI_Irecv(&ipc_handles_prev[dir],
                                 sizeof(cudaIpcEventHandle_t), MPI_BYTE,
                                 peer_id, 0, m_comm, &req[req_idx++]));
        cudaIpcEventHandle_t local_event_h;
        COLL_CHECK_CUDA(cudaIpcGetEventHandle(&local_event_h,
                                              comp_ev(dir)));
        COLL_CHECK_MPI(MPI_Isend(&local_event_h,
                                 sizeof(cudaIpcEventHandle_t), MPI_BYTE,
                                 peer_id, 0, m_comm, &req[req_idx++]));
      }
    }
    COLL_CHECK_MPI(MPI_Waitall(req_idx, req, MPI_STATUS_IGNORE));
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;
      if (next_access_type(dir) != MPI) {
        COLL_CHECK_CUDA(cudaIpcOpenEventHandle(&next_ev(dir),
                                               ipc_handles_next[dir]));
      }
      if (prev_access_type(dir) != MPI) {
        COLL_CHECK_CUDA(cudaIpcOpenEventHandle(&prev_ev(dir),
                                               ipc_handles_prev[dir]));
      }
    }
  }

  void destroy_inter_process_events() {
    for (TransferDir dir: DIRECTIONS) {
      if (!m_trans_dir[dir]) continue;      
      if (next_access_type(dir) != MPI) {
        COLL_CHECK_CUDA(cudaEventDestroy(next_ev(dir)));
      }
      if (prev_access_type(dir) != MPI) {
        COLL_CHECK_CUDA(cudaEventDestroy(prev_ev(dir)));
      }
    }
  }

  void setup_remote_buffer_mapping(TransferDir dir) {
    if (!m_trans_dir[dir]) return;
    cudaIpcMemHandle_t local_ipc_h, peer_ipc_h;
    // If the peer can access the local device with CUDA API, expose its buffer
    // address as an IPC handle
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
      change_device_to_accessible_to_next(dir);
      COLL_CHECK_CUDA(cudaIpcOpenMemHandle(&next_work(dir), peer_ipc_h,
                                           cudaIpcMemLazyEnablePeerAccess));
      reset_device();
    }
  }

  void close_remote_buffer_mapping(TransferDir dir) {
    if (next_work(dir) != nullptr) {
      change_device_to_accessible_to_next(dir);
      COLL_CHECK_CUDA(cudaIpcCloseMemHandle(next_work(dir)));
      next_work(dir) = nullptr;
      reset_device();
    }
  }

  void close_remote_buffer_mapping() {
    for (TransferDir dir: DIRECTIONS) {
      close_remote_buffer_mapping(dir);
    }
  }

  size_t setup_pe_counts(size_t count, std::vector<size_t> *pe_counts,
                         std::vector<size_t> *pe_offsets) {
    int num_directions = m_trans_dir[R2L] ? 2 : 1;
    size_t pe_count_base = count / (m_np * num_directions);
    int rem = count % (m_np * num_directions);
    pe_offsets[0].push_back(0);
    for (TransferDir dir: DIRECTIONS) {
      for (int j = 0; j < m_np; ++j) {
        int idx = j + (dir == R2L ? m_np : 0);
        pe_counts[dir].push_back(pe_count_base + ((idx < rem) ? 1 : 0));
        if (j > 0) {
          pe_offsets[dir].push_back(pe_offsets[dir][j-1] + pe_counts[dir][j-1]);
        } else if (dir == R2L) {
          pe_offsets[dir].push_back(pe_offsets[L2R][m_np-1] + pe_counts[L2R][m_np-1]);
        }
      }
    }
    size_t max_count = pe_count_base + (rem > 0 ? 1 : 0);
    return max_count;
  }

  void notify(int rank, int tag) {
    char x = 0;
    COLL_CHECK_MPI(MPI_Send(
        &x, 1, MPI_CHAR, rank, tag, m_comm));
  }

  void notify_next_rank(TransferDir dir) {
    notify(next_pid(dir), notification_next_tag(dir));
  }
  
  void notify_prev_rank(TransferDir dir) {
    notify(prev_pid(dir), notification_prev_tag(dir));
  }

  void wait_for_notification(int rank, int tag) {
    char x;
    COLL_CHECK_MPI(MPI_Recv(
        &x, 1, MPI_CHAR, rank, tag,
        m_comm, MPI_STATUS_IGNORE));
  }

  void wait_for_prev_rank(TransferDir dir) {
    wait_for_notification(prev_pid(dir), notification_next_tag(dir));
  }

  void wait_for_next_rank(TransferDir dir) {
    wait_for_notification(next_pid(dir), notification_prev_tag(dir));
  }

  template <typename T>
  void transfer(T *buf, T *work_buf,
                size_t send_count, size_t recv_count,
                size_t send_offset,
                cudaStream_t stream, int iter_idx,
                MPI_Request *requests, int &num_requests,
                const TransferDir dir) {
    const int tag = 0;
    const MPI_Datatype mpi_type = get_mpi_data_type<T>();
    num_requests = 0;
    if (prev_access_type(dir) == MPI) {
      COLL_CHECK_CUDA(cudaEventSynchronize(comp_ev(dir)));
      COLL_CHECK_MPI(MPI_Irecv(
          work_buf, recv_count, mpi_type, prev_pid(dir), tag,
          m_comm, &requests[num_requests++]));
    }
    // Send to neighbor        
    T *src_ptr = buf + send_offset;
    if (next_access_type(dir) == MPI) {
      if (prev_access_type(dir) != MPI) {
        COLL_CHECK_CUDA(cudaEventSynchronize(comp_ev(dir)));
      }
      COLL_CHECK_MPI(MPI_Isend(
          src_ptr, send_count, mpi_type,
          next_pid(dir), tag, m_comm, &requests[num_requests++]));
    } else {
      // Make sure the completion event of computation was
      // recorded before sending the buffer
      if (iter_idx != 0) {
        wait_for_next_rank(dir);
      } 
      COLL_CHECK_CUDA(cudaStreamWaitEvent(
          stream, next_ev(dir), 0));
      COLL_CHECK_CUDA(cudaMemcpyPeerAsync(
          next_work(dir), next_dev(dir), src_ptr, m_gpu,
          send_count * sizeof(T), stream));
      COLL_CHECK_CUDA(cudaEventRecord(trans_ev(dir), stream));
      notify_next_rank(dir);
    }
  }

  void ensure_transfer(MPI_Request *requests, int num_requests,
                       cudaStream_t stream) {
    // Set dependency for transfer from the neighbor device
    for (TransferDir dir: DIRECTIONS) {
      if (m_trans_dir[dir] && prev_access_type(dir) != MPI) {
        wait_for_prev_rank(dir);
        COLL_CHECK_CUDA(cudaStreamWaitEvent(
            stream, prev_ev(dir), 0));
      }
    }
    // Wait for completion of MPI transfer
    COLL_CHECK_MPI(MPI_Waitall(num_requests, requests, MPI_STATUS_IGNORE));
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
                cudaStream_t stream, bool bidirectional=true) {
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

    // Step 1: Reduce-scatter
    // Step 2: Allgather
    for (int step = 0; step < 2; ++step) {    
      for (int i = 0; i < m_np - 1; ++i) {
        MPI_Request requests[4];
        int num_requests = 0;
        for (TransferDir dir: DIRECTIONS) {
          if (!m_trans_dir[dir]) continue;          
          int nr;
          transfer(buf, work_bufs[dir],
                   pe_counts[dir][send_idx[dir]],
                   pe_counts[dir][recv_idx[dir]],
                   pe_offsets[dir][send_idx[dir]],
                   stream, i, requests + num_requests, nr, dir);
          num_requests += nr;
        }
        ensure_transfer(requests, num_requests, stream);
        for (TransferDir dir: DIRECTIONS) {
          if (!m_trans_dir[dir]) continue;
          if (step == 0) {
            reduce1(buf + pe_offsets[dir][recv_idx[dir]],
                    work_bufs[dir],
                    pe_counts[dir][recv_idx[dir]],
                    stream, op,
                    GetReductionOperandType<T>::key);
          } else {
            COLL_CHECK_CUDA(cudaMemcpyAsync(
                buf + pe_offsets[dir][recv_idx[dir]],
                work_bufs[dir],
                pe_counts[dir][recv_idx[dir]] * sizeof(T),
                cudaMemcpyDeviceToDevice, stream));
          }
          COLL_CHECK_CUDA(cudaEventRecord(comp_ev(dir), stream));
          // notify prev rank for the recording of comp event if MPI
          // is not used
          if (prev_access_type(dir) != MPI) notify_prev_rank(dir);
        }
        update_indices(send_idx, recv_idx);
      }
      // There remains an un-received notificaton
      for (TransferDir dir: DIRECTIONS) {
        if (!m_trans_dir[dir]) continue;
        if (next_access_type(dir) != MPI) {
          wait_for_next_rank(dir);
        }
      }
    }
                                        
    return 0;
  }

 protected:

  int &rid() {
    return m_rid;
  }

  int &pid(Side s) {
    if (s == LHS) {
      return m_pid_lhs;
    } else {
      return m_pid_rhs;
    }
  }

  int &prev_pid(TransferDir dir) {
    return pid(prev_side(dir));
  }

  int &next_pid(TransferDir dir) {
    return pid(next_side(dir));
  }

  AccessType &access_type(Side s) {
    return m_access_type[s];
  }

  AccessType &prev_access_type(TransferDir dir) {
    return access_type(prev_side(dir));
  }

  AccessType &next_access_type(TransferDir dir) {
    return access_type(next_side(dir));
  }

  int &neighbor_dev(Side s) {
    return m_neighbor_dev[s];
  }

  int &next_dev(TransferDir dir) {
    return neighbor_dev(next_side(dir));
  }

  int &prev_dev(TransferDir dir) {
    return neighbor_dev(prev_side(dir));
  }

  Side prev_side(TransferDir trans_dir) {
    return trans_dir == L2R ? LHS : RHS;
  }

  Side next_side(TransferDir trans_dir) {
    return trans_dir == L2R ? RHS : LHS;
  }

  void *&next_work(TransferDir dir) {
    return m_neighbor_work[dir];
  }

  cudaEvent_t &comp_ev(TransferDir dir) {
    return m_ev_comp[dir];
  }

  cudaEvent_t &trans_ev(TransferDir dir) {
    return m_ev_trans[dir];
  }

  cudaEvent_t &next_ev(TransferDir dir) {
    return m_next_proc_ev[dir];
  }

  cudaEvent_t &prev_ev(TransferDir dir) {
    return m_prev_proc_ev[dir];
  }

  int notification_next_tag(TransferDir dir) {
    return m_notification_next_tags[dir];
  }

  int notification_prev_tag(TransferDir dir) {
    return m_notification_prev_tags[dir];
  }

  // current device setting
  void reset_device() {
    COLL_CHECK_CUDA(cudaSetDevice(m_gpu));
  }

  void change_device_to_accessible_to_next(TransferDir dir) {
    // If it needs to be accessed through host memory, open the
    // IPC handle at a context on the remote GPU
    if (next_access_type(dir) == HOST) {
#ifdef AL_MPI_CUDA_DEBUG
      MPIPrintStream(std::cerr, m_pid)()
          << "Opening a context on a remote GPU, "
          << get_neighbor_dev(put_dst) << ", from process " << m_pid << "\n";
#endif
      COLL_CHECK_CUDA(cudaSetDevice(next_dev(dir)));
    }
  }

  int m_gpu;
  
  MPI_Comm m_comm;  
  void *m_gpu_bufs[2] = {nullptr, nullptr};
  size_t m_gpu_buf_sizes[2] = {0, 0};

  int m_np;
  int m_pid;
  int m_rid;
  int m_pid_lhs;
  int m_pid_rhs;
  int m_send_idx[2];
  int m_recv_idx[2];

  cudaEvent_t m_ev_comp[2];
  cudaEvent_t m_ev_trans[2];

  AccessType m_access_type[2];
  int m_neighbor_dev[2];
  void *m_neighbor_work[2] = {nullptr, nullptr};
  cudaEvent_t m_prev_proc_ev[2];
  cudaEvent_t m_next_proc_ev[2];

  bool m_trans_dir[2] = {true, true};

  int m_notification_next_tags[2] = {1001, 1002};
  int m_notification_prev_tags[2] = {1003, 1004};
};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
