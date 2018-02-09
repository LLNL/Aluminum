#pragma once

#include "mpi_cuda/util.hpp"
#include "mpi_cuda/cuda_kernels.hpp"
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

namespace allreduces {
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
  enum AccessType {MPI, PEER, HOST};
  enum AccessDir {LHS=0, RHS=1};
  
 public:
  RingMPICUDA(MPI_Comm comm): m_comm(comm) {
    int device = -1;
    COLL_CHECK_CUDA(cudaGetDevice(&device));
    m_gpus = {device};
    m_num_gpus = 1;
    build_ring();
    setup_events();
  }
  
  RingMPICUDA(const std::vector<int> gpus,
              MPI_Comm comm): m_gpus(gpus), m_comm(comm) {
    m_num_gpus = m_gpus.size();
    build_ring();
    setup_events();
  }

  RingMPICUDA(int num_gpus, MPI_Comm comm): m_num_gpus(num_gpus), m_comm(comm) {
    for (int i = 0; i < num_gpus; ++i) {
      m_gpus.push_back(i);
    }
    build_ring();
    setup_events();
  }
  
  ~RingMPICUDA() {
    destroy_events();
    close_remote_buffer_mapping(m_neighbor_buf);
    close_remote_buffer_mapping(m_neighbor_work);
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
#ifdef ALUMINUM_MPI_CUDA_DEBUG
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
#ifdef ALUMINUM_MPI_CUDA_DEBUG      
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
  void get_gpu_bufs(size_t count, std::vector<T*> *bufs) {
    size_t real_size = sizeof(T) * count;
    //if (m_trans_dir[R2L]) real_size *= 2;
    if (!(m_gpu_bufs[L2R].size() > 0 && m_gpu_buf_size >= real_size)) {
      close_remote_buffer_mapping(m_neighbor_work);
      // Make sure everyone unmap the IPC handle before actually
        // freeing the memory
      COLL_CHECK_MPI(MPI_Barrier(m_comm));
      free_gpu_bufs();
      for (int i = 0; i < m_num_gpus; ++i) {
        COLL_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
        void *p = nullptr;
        COLL_CHECK_CUDA(cudaMalloc(&p, real_size));
        COLL_ASSERT(p != nullptr);
        m_gpu_bufs[L2R].push_back(p);
        if (m_trans_dir[R2L]) {
          COLL_CHECK_CUDA(cudaMalloc(&p, real_size));
          COLL_ASSERT(p != nullptr);
          m_gpu_bufs[R2L].push_back(static_cast<T*>(p));
        }
        m_gpu_buf_size = real_size;
        setup_remote_buffer_mapping(&m_gpu_bufs[L2R], &m_gpu_bufs[R2L],
                                    m_neighbor_work);
      }
    }
    for (int i = 0; i < m_num_gpus; ++i) {
      bufs[L2R].push_back(static_cast<T*>(m_gpu_bufs[L2R][i]));
      if (m_trans_dir[R2L]) {
        bufs[R2L].push_back(static_cast<T*>(m_gpu_bufs[R2L][i]));
      }
    }
    return;
  }

  void free_gpu_bufs() {
    if (m_gpu_bufs[L2R].size() > 0) {
      for (int i = 0; i < m_num_gpus; ++i) {
        COLL_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
        COLL_CHECK_CUDA(cudaFree(m_gpu_bufs[L2R][i]));
        if (m_trans_dir[R2L])
          COLL_CHECK_CUDA(cudaFree(m_gpu_bufs[R2L][i]));
      }
      m_gpu_bufs[L2R].clear();
      m_gpu_bufs[R2L].clear();        
    }
  }

  void setup_events() {
    //m_ev_comp = new cudaEvent_t[m_num_gpus];
    //m_ev_trans = new cudaEvent_t[m_num_gpus];    
    for (int i = 0; i < m_num_gpus; ++i) {
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      for (int dir = 0; dir < 2; ++dir) {
        if (!m_trans_dir[dir]) continue;
        cudaEvent_t ev;
        unsigned flag = cudaEventDisableTiming;
        if ((i == 0 && dir == L2R && m_access_type[LHS] != MPI) ||
            (i == m_num_gpus - 1 && dir == R2L && m_access_type[RHS] != MPI)) {
          flag |= cudaEventInterprocess;
        }
        COLL_CHECK_CUDA(cudaEventCreateWithFlags(&ev, flag));
        m_ev_comp[dir].push_back(ev);
        flag = cudaEventDisableTiming;
        if ((i == m_num_gpus - 1 && dir == L2R && m_access_type[RHS] != MPI) ||
            (i == 0 && dir == R2L && m_access_type[LHS] != MPI)) {
          flag |= cudaEventInterprocess;
        }
        COLL_CHECK_CUDA(cudaEventCreateWithFlags(&ev, flag));
        m_ev_trans[dir].push_back(ev);
      }
    }
    setup_inter_process_events();
  }

  void destroy_events() {
    destroy_inter_process_events();
    // make sure exported events are first destroyed at remote
    // processes, though it may not be necessary.
    COLL_CHECK_MPI(MPI_Barrier(m_comm));
    for (int i = 0; i < m_num_gpus; ++i) {
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus[i]));
      for (int dir = 0; dir < 2; ++dir) {
        if (!m_trans_dir[dir]) continue;        
        COLL_CHECK_CUDA(cudaEventDestroy(m_ev_comp[dir][i]));      
        COLL_CHECK_CUDA(cudaEventDestroy(m_ev_trans[dir][i]));
      }
    }
  }

  void build_ring() {
    COLL_CHECK_MPI(MPI_Comm_size(m_comm, &m_np));
    COLL_CHECK_MPI(MPI_Comm_rank(m_comm, &m_pid));  
    get_ring_indices(m_pid, m_np, m_rid, m_rid_lhs, m_rid_rhs);
    int num_total_gpus = m_np * m_num_gpus;
    for (int i = 0; i < m_num_gpus; ++i) {
      m_gpu_rid.push_back(m_rid * m_num_gpus + i);
      for (int dir = 0; dir < 2; ++dir) {
        m_send_idx[dir].push_back(m_gpu_rid[i]);
        int recv_idx = dir == L2R ?
            dec(m_send_idx[dir][i], num_total_gpus) :
            inc(m_send_idx[dir][i], num_total_gpus);
        m_recv_idx[dir].push_back(recv_idx);
#if 0        
        MPIPrintStream(std::cout, m_pid)() <<
            "send_idx[" << dir << "][" << i << "]="
                        << m_send_idx[dir][i] << "\n";
        MPIPrintStream(std::cout, m_pid)() <<
            "recv_idx[" << dir << "][" << i << "]="
                        << m_recv_idx[dir][i] << "\n";
#endif        
      }
    }

    // exchange device id with RHS
    int dev1 = m_gpus[0];
    int dev2 = m_gpus[m_num_gpus-1];
    COLL_CHECK_MPI(MPI_Sendrecv(
        &dev2, 1, MPI_INT, m_rid_rhs, 0,
        &m_neighbor_dev[0], 1, MPI_INT, m_rid_lhs, 0,
        m_comm, MPI_STATUS_IGNORE));
    COLL_CHECK_MPI(MPI_Sendrecv(
        &dev1, 1, MPI_INT, m_rid_lhs, 0,
        &m_neighbor_dev[1], 1, MPI_INT, m_rid_rhs, 0,
        m_comm, MPI_STATUS_IGNORE));
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
        proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, m_rid_rhs, 0,
        proc_name_lhs, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, m_rid_lhs, 0,
        m_comm, MPI_STATUS_IGNORE));
    COLL_CHECK_MPI(MPI_Sendrecv(
        proc_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, m_rid_lhs, 0,
        proc_name_rhs, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, m_rid_rhs, 0,
        m_comm, MPI_STATUS_IGNORE));

    // Check whether the neighbor devices can be accessed with CUDA
    // API.
    for (int i = 0; i < 2; ++i) {
      int local_dev_id = i == 0 ? 0 : m_num_gpus - 1;
      int local_dev = m_gpus[local_dev_id];
      char *peer_proc_name = i == 0 ? proc_name_lhs : proc_name_rhs;
      if (std::strcmp(peer_proc_name, proc_name) == 0) {
        int peer_access = 0;
        std::cerr << "[" << m_pid << "] enable peer; local_dev: " << local_dev
                  << ", peer dev: " << m_neighbor_dev[i] << "\n";
        COLL_CHECK_CUDA(cudaSetDevice(local_dev));
        COLL_CHECK_CUDA(cudaDeviceCanAccessPeer(&peer_access, local_dev, m_neighbor_dev[i]));
        if (peer_access) {
          cudaError_t err = cudaDeviceEnablePeerAccess(m_neighbor_dev[i], 0);
          if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            std::cerr << "Enabling peer access failed\n";
            abort();
          }
          m_access_type[i] = PEER;        
        } else {
          m_access_type[i] = HOST;        
        }
      } else {
        m_access_type[i] = MPI;
      }
    }
  }

  void setup_inter_process_events() {
    // exchange inter-process events
    MPI_Request req[8];
    int req_idx = 0;
    //cudaIpcEventHandle_t lhs_event_h, rhs_event_h;
    cudaIpcEventHandle_t ipc_handles[2][2]; // [DIR][NI]
    for (int dir = 0; dir < 2; ++dir) { // L2R and R2L
      if (!m_trans_dir[dir]) continue;
      for (int ni = 0; ni < 2; ++ni) { // LHS and RHS
        if (m_access_type[ni] != MPI) {
          int peer_id = ni == 0 ? m_rid_lhs : m_rid_rhs;
          COLL_CHECK_MPI(MPI_Irecv(&ipc_handles[dir][ni],
                                   sizeof(cudaIpcEventHandle_t), MPI_BYTE,
                                   peer_id, 0, m_comm, &req[req_idx++]));
          int dev_id = ni == LHS ? m_gpus.front() : m_gpus.back();
          cudaIpcEventHandle_t local_event_h;
          cudaEvent_t *local_event = nullptr;
          if (dir == L2R && ni == LHS) {
            local_event = &m_ev_comp[dir].front();
          } else if (dir == L2R && ni == RHS) {
            local_event = &m_ev_trans[dir].back();
          } else if (dir == R2L && ni == LHS) {
            local_event = &m_ev_trans[dir].front();            
          } else if (dir == R2L && ni == RHS) {
            local_event = &m_ev_comp[dir].back();            
          }
          COLL_CHECK_CUDA(cudaSetDevice(dev_id));
          COLL_CHECK_CUDA(cudaIpcGetEventHandle(&local_event_h,
                                                *local_event));
          COLL_CHECK_MPI(MPI_Isend(&local_event_h,
                                   sizeof(cudaIpcEventHandle_t), MPI_BYTE,
                                   peer_id, 0, m_comm, &req[req_idx++]));
        }
      }
    }
    COLL_CHECK_MPI(MPI_Waitall(req_idx, req, MPI_STATUS_IGNORE));
    for (int dir = 0; dir < 2; ++dir) { // L2R and R2L
      if (!m_trans_dir[dir]) continue;      
      for (int ni = 0; ni < 2; ++ni) { // LHS and RHS
        if (m_access_type[ni] != MPI) {
          int dev_id = ni == LHS ? m_gpus.front() : m_gpus.back();
          COLL_CHECK_CUDA(cudaSetDevice(dev_id));
          COLL_CHECK_CUDA(cudaIpcOpenEventHandle(&m_neighbor_ev[dir][ni],
                                                 ipc_handles[dir][ni]));
        }
      }
    }
  }

  void destroy_inter_process_events() {
    for (int dir = 0; dir < 2; ++dir) { // L2R and R2L
      if (!m_trans_dir[dir]) continue;      
      for (int ni = 0; ni < 2; ++ni) { // LHS and RHS
        if (m_access_type[ni] != MPI) {
          int dev_id = ni == LHS ? m_gpus.front() : m_gpus.back();
          COLL_CHECK_CUDA(cudaSetDevice(dev_id));
          COLL_CHECK_CUDA(cudaEventDestroy(m_neighbor_ev[dir][ni]));
        }
      }
    }
  }

  template <typename T>
  void setup_remote_buffer_mapping(const std::vector<T*> *bufs_l2r,
                                   const std::vector<T*> *bufs_r2l,
                                   void **neighbor_ptrs) {
    // i:0 -> lhs, i:1 -> rhs
    for (int i = 0; i < 2; ++i) {
      int accessed_dir = i^1;
      int transfer_dir = i^1;
      if (!m_trans_dir[transfer_dir]) continue;
      int dev_accessed_by_peer_id = i == LHS ? m_num_gpus - 1 : 0;
      int dev_accessed_by_peer = m_gpus[dev_accessed_by_peer_id];
      int dev_accessing_peer_id = i == LHS ? 0 : m_num_gpus - 1;
      int dev_accessing_peer = m_gpus[dev_accessing_peer_id];
      //int local_dev = m_gpus[local_dev_id];
      cudaIpcMemHandle_t local_ipc_h, peer_ipc_h;
      // If the peer can access the local device with CUDA API, expose its buffer
      // address as an IPC handle
      int send_pid = i == 0 ? m_rid_rhs : m_rid_lhs;
      int recv_pid = i == 0 ? m_rid_lhs : m_rid_rhs;
      size_t send_msg_size = 0;
      if (m_access_type[accessed_dir] != MPI) {
        COLL_CHECK_CUDA(cudaSetDevice(dev_accessed_by_peer));
        const std::vector<T*> *bufs = transfer_dir == L2R ? bufs_l2r : bufs_r2l;
#ifdef COLL_DEBUG        
        MPIPrintStream(std::cerr, m_pid)()
            << "exposing " << bufs->at(dev_accessed_by_peer_id)
            << "  to " << send_pid << "\n";
#endif        
        COLL_CHECK_CUDA(cudaIpcGetMemHandle(
            &local_ipc_h, bufs->at(dev_accessed_by_peer_id)));
        send_msg_size = sizeof(cudaIpcMemHandle_t);
      }
      size_t recv_msg_size =
          m_access_type[i] != MPI ? sizeof(cudaIpcMemHandle_t) : 0;
      COLL_CHECK_MPI(MPI_Sendrecv(
          &local_ipc_h, send_msg_size,  MPI_BYTE, send_pid, 0,
          &peer_ipc_h, recv_msg_size, MPI_BYTE, recv_pid, 0,
          m_comm, MPI_STATUS_IGNORE));
      if (m_access_type[i] != MPI) {
        // If it needs to be accessed through host memory, open the
        // IPC handle at a context on the remote GPU
        if (m_access_type[i] == HOST) {
          COLL_CHECK_CUDA(cudaSetDevice(m_neighbor_dev[i]));
        } else {
          COLL_CHECK_CUDA(cudaSetDevice(dev_accessing_peer));
        }
        COLL_CHECK_CUDA(cudaIpcOpenMemHandle(&(neighbor_ptrs[i]), peer_ipc_h,
                                             cudaIpcMemLazyEnablePeerAccess));
      } else {
        //std::cerr << "[" << m_pid << "] Remote dev is accessed with MPI\n";
      }
    }
  }

  void close_remote_buffer_mapping(void **neighbor_ptrs) {
    for (int i = 0; i < 2; ++i) {
      if (neighbor_ptrs[i] != nullptr) {
        if (m_access_type[i] == HOST) {
          COLL_CHECK_CUDA(cudaSetDevice(m_neighbor_dev[i]));
        } else {
          COLL_CHECK_CUDA(cudaSetDevice(m_gpus[i == 0 ? 0 : m_num_gpus - 1]));
        }
        COLL_CHECK_CUDA(cudaIpcCloseMemHandle(neighbor_ptrs[i]));
        neighbor_ptrs[i] = nullptr;
      }
    }
  }

  template <typename T>
  void setup_remote_buffer_mapping_with_caching(
      const std::vector<T*> *bufs_l2r,
      const std::vector<T*> *bufs_r2l) {
    // Use the pointer value of the first device as the key for
    // caching 
    if (m_ipc_cache_buf == bufs_l2r->front()) {
      // Can reuse the previous mapping
#ifdef ALUMINUM_MPI_CUDA_DEBUG
      MPIPrintStream(std::cerr, m_pid)() << "Reusing remote buffer mapping\n";
#endif      
      return;
    }
    // Close the current mapping if exists
    if (m_ipc_cache_buf != nullptr) {
#ifdef ALUMINUM_MPI_CUDA_DEBUG
      MPIPrintStream(std::cerr, m_pid)() << "Destroy previous mapping\n";
#endif      
      close_remote_buffer_mapping(m_neighbor_buf);
    }
#ifdef ALUMINUM_MPI_CUDA_DEBUG
    MPIPrintStream(std::cerr, m_pid)() << "setup new mapping\n";
#endif      
    setup_remote_buffer_mapping(bufs_l2r, bufs_r2l, m_neighbor_buf);
    m_ipc_cache_buf = bufs_l2r->front();
  }


  void notify(int rank, int dir) {
    char x = 0;
    COLL_CHECK_MPI(MPI_Send(&x, 1, MPI_CHAR, rank,
                            m_notification_tags[dir], m_comm));
  }

  void notify_next_rank(int dir) {
    int pid = dir == L2R ? m_rid_rhs : m_rid_lhs;
    int dev_known_to_mpi = dir == L2R ?
        m_gpus.back() : m_gpus.front();
    int current_device;
    COLL_CHECK_CUDA(cudaGetDevice(&current_device));
    COLL_CHECK_CUDA(cudaSetDevice(dev_known_to_mpi));
    notify(pid, dir);
    COLL_CHECK_CUDA(cudaSetDevice(current_device));        
  }
  
  void notify_prev_rank(int dir) {
    int pid = dir == L2R ? m_rid_lhs : m_rid_rhs;
    int dev_known_to_mpi = dir == L2R ?
        m_gpus.front() : m_gpus.back();
    int current_device;
    COLL_CHECK_CUDA(cudaGetDevice(&current_device));
    COLL_CHECK_CUDA(cudaSetDevice(dev_known_to_mpi));
    notify(pid, dir);
    COLL_CHECK_CUDA(cudaSetDevice(current_device));        
  }

  void wait_for_notification(int rank, int dir) {
    char x;
    COLL_CHECK_MPI(MPI_Recv(&x, 1, MPI_CHAR, rank, m_notification_tags[dir],
                            m_comm, MPI_STATUS_IGNORE));
  }

  void wait_for_prev_rank(int dir) {
    int pid = dir == L2R ? m_rid_lhs : m_rid_rhs;
    int dev_known_to_mpi = dir == L2R ?
        m_gpus.front() : m_gpus.back();
    int current_device;
    COLL_CHECK_CUDA(cudaGetDevice(&current_device));
    COLL_CHECK_CUDA(cudaSetDevice(dev_known_to_mpi));
    wait_for_notification(pid, dir);
    COLL_CHECK_CUDA(cudaSetDevice(current_device));        
  }

  void wait_for_next_rank(int dir) {
    int pid = dir == L2R ? m_rid_rhs : m_rid_lhs;
    int dev_known_to_mpi = dir == L2R ?
        m_gpus.back() : m_gpus.front();
    int current_device;
    COLL_CHECK_CUDA(cudaGetDevice(&current_device));
    COLL_CHECK_CUDA(cudaSetDevice(dev_known_to_mpi));
    wait_for_notification(pid, dir);
    COLL_CHECK_CUDA(cudaSetDevice(current_device));        
  }

  void ensure_transfer(MPI_Request *requests, int num_requests,
                       std::vector<cudaStream_t> &streams) {
    // Set dependency for transfer from the neighbor device
    if (m_trans_dir[L2R] && m_access_type[LHS] != MPI) {
      wait_for_prev_rank(L2R);
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus.front()));      
      COLL_CHECK_CUDA(cudaStreamWaitEvent(streams.front(),
                                          m_neighbor_ev[L2R][LHS], 0));
    }
    if (m_trans_dir[R2L] && m_access_type[RHS] != MPI) {
      wait_for_prev_rank(R2L);
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus.back()));      
      COLL_CHECK_CUDA(cudaStreamWaitEvent(streams.back(),
                                          m_neighbor_ev[R2L][RHS], 0));
    }
    // Wait for completion of MPI transfer
    if (m_access_type[LHS] == MPI) {
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus.front()));
    } else {
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus.back()));
    }
    COLL_CHECK_MPI(MPI_Waitall(num_requests, requests, MPI_STATUS_IGNORE));
  }

  template <typename T>
  void transfer(const std::vector<T*> &bufs,
                const std::vector<T*> &work_bufs,
                const std::vector<size_t> *pe_counts,
                const std::vector<size_t> *pe_offsets,                 
                std::vector<cudaStream_t> &streams,
                int iter_idx,
                MPI_Request *requests,
                int &num_requests,
                const TransferDir trans,
                bool scatter_reduce) {
    const int tag = 0;
    const MPI_Datatype mpi_type = get_mpi_data_type<T>();
    const AccessDir src = get_src_dir(trans);    
    const AccessDir dst = get_dst_dir(trans);
    const bool src_require_mpi = m_access_type[src] == MPI;        
    const bool dst_require_mpi = m_access_type[dst] == MPI;
    const int src_mpi_rank = trans == L2R ? m_rid_lhs : m_rid_rhs;
    const int dst_mpi_rank = trans == L2R ? m_rid_rhs : m_rid_lhs;    
    const int first_dev = trans == L2R ? 0 : m_num_gpus - 1;
    const int last_dev = trans == L2R ? m_num_gpus - 1 : 0;
    num_requests = 0;
    for (int g = first_dev; true; g += (trans == L2R ? 1 : -1)) {
      COLL_CHECK_CUDA(cudaSetDevice(m_gpus[g]));
      size_t send_offset = pe_offsets[trans][m_send_idx[trans][g]];
      size_t recv_offset = pe_offsets[trans][m_recv_idx[trans][g]]; 
      size_t send_count = pe_counts[trans][m_send_idx[trans][g]];
      size_t recv_count = pe_counts[trans][m_recv_idx[trans][g]]; 
      if (g == first_dev && src_require_mpi) {
        if (scatter_reduce) {
          COLL_CHECK_CUDA(cudaEventSynchronize(m_ev_comp[trans][g]));
        }
        T *recv_ptr = scatter_reduce ? work_bufs[g] : bufs[g] + recv_offset;
        COLL_CHECK_MPI(MPI_Irecv(
            recv_ptr, recv_count, mpi_type, src_mpi_rank, tag,
            m_comm, &requests[num_requests++]));
      }
      if (g == last_dev) {
        if (scatter_reduce) {
          if (dst_require_mpi) {
            COLL_CHECK_CUDA(cudaEventSynchronize(m_ev_comp[trans][g]));
          } else {
            // Make sure the completion event of computation was
            // recorded before sending the buffer
            if (iter_idx != 0) {
              wait_for_next_rank(trans);
            } 
            COLL_CHECK_CUDA(cudaStreamWaitEvent(
                streams[g], m_neighbor_ev[trans][dst], 0));
          }
        } else {
          if (dst_require_mpi) {
            // wait for the completion of scatter reduce or transfer
            // from previous device
            COLL_CHECK_CUDA(cudaEventSynchronize(m_ev_comp[trans][g]));
          }
        }
        // Send to neighbor        
        T *src_ptr = bufs[g] + send_offset;
        if (dst_require_mpi) {
          COLL_CHECK_MPI(MPI_Isend(
              src_ptr, send_count, mpi_type,
              dst_mpi_rank, tag, m_comm, &requests[num_requests++]));
        } else {
          void *dst_ptr = scatter_reduce ? m_neighbor_work[dst] :
              static_cast<T*>(m_neighbor_buf[dst]) + send_offset;
          COLL_CHECK_CUDA(cudaMemcpyPeerAsync(
              dst_ptr, m_neighbor_dev[dst], src_ptr, g,
              send_count * sizeof(T), streams[g]));
          COLL_CHECK_CUDA(cudaEventRecord(m_ev_trans[trans][g], streams[g]));
          notify_next_rank(trans);
        }
      }
      if (g != last_dev) {
        int next_dev = trans == L2R ? g+1 : g-1;
        if (scatter_reduce) {
          COLL_CHECK_CUDA(cudaStreamWaitEvent(streams[g], m_ev_comp[trans][next_dev], 0));
        }
        T *src_ptr = bufs[g] + send_offset;
        T *dst_ptr = scatter_reduce ? work_bufs[next_dev] : bufs[next_dev] + send_offset;
        COLL_CHECK_CUDA(cudaMemcpyPeerAsync(
            dst_ptr, next_dev, src_ptr, g, send_count * sizeof(T),
            streams[g]));
        COLL_CHECK_CUDA(cudaEventRecord(m_ev_trans[trans][g], streams[g]));
      }
      if (g != first_dev) {
        int prev_dev = trans == L2R ? g-1 : g+1;        
        COLL_CHECK_CUDA(cudaStreamWaitEvent(streams[g], m_ev_trans[trans][prev_dev], 0));
        if (scatter_reduce) {
          T *dst_ptr = bufs[g] + recv_offset;
          reduce1(dst_ptr, work_bufs[g], recv_count, streams[g]);
          COLL_CHECK_CUDA(cudaEventRecord(m_ev_comp[trans][g], streams[g]));
        } else {
          // Record the event so that the send op can wait for the
          // completion of transfer
          COLL_CHECK_CUDA(cudaEventRecord(m_ev_comp[trans][g], streams[g]));
        }
      }
      if (g == last_dev) break;      
    }
  }
  
  size_t setup_pe_counts(size_t count, std::vector<size_t> *pe_counts,
                         std::vector<size_t> *pe_offsets) {
    int num_total_gpus = m_np * m_num_gpus;
    int num_directions = m_trans_dir[R2L] ? 2 : 1;
    size_t pe_count_base = count / (num_total_gpus * num_directions);
    int rem = count % (num_total_gpus * num_directions);
    pe_offsets[0].push_back(0);
    for (int i = 0; i < num_directions; ++i) {
      for (int j = 0; j < num_total_gpus; ++j) {
        pe_counts[i].push_back(pe_count_base + ((i * num_total_gpus + j < rem) ? 1 : 0));
        if (j > 0) {
          pe_offsets[i].push_back(pe_offsets[i][j-1] + pe_counts[i][j-1]);
        } else if (i > 0) {
          pe_offsets[i].push_back(pe_offsets[i-1][num_total_gpus-1] + pe_counts[i-1][num_total_gpus-1]);
        }
      }
    }
    size_t max_count = pe_count_base + (rem > 0 ? 1 : 0);
    return max_count;
  }
  
 public:
  // bufs: GPU buffers
  template <typename T>
  int allreduce(const std::vector<T*> &bufs,
                size_t count,
                std::vector<cudaStream_t> *streams=nullptr,
                bool bidirectional=true) {
    if (count == 0) return 0;
    
    int num_total_gpus = m_np * m_num_gpus;
    std::vector<size_t> pe_counts[2];
    std::vector<size_t> pe_offsets[2];
    const size_t max_pe_count = setup_pe_counts(count, pe_counts, pe_offsets);

    //std::cout << "pe_count (bi): " << pe_count << std::endl;    

    std::vector<T*> work_bufs[2];
    get_gpu_bufs<T>(max_pe_count, work_bufs);

    // Keep the indices for restoring them when allreduce is done
    std::vector<int> send_idx_ref[2] = {m_send_idx[0], m_send_idx[1]};
    std::vector<int> recv_idx_ref[2] = {m_recv_idx[0], m_recv_idx[1]};

    bool stream_passed = true;

    if (!streams) {
      stream_passed = false;
      streams = new std::vector<cudaStream_t>();
      create_streams(*streams, m_gpus);      
    }

    // Map the base addresses as IPC handles can be taken only for
    // base addresses.  
    setup_remote_buffer_mapping_with_caching<T>(&bufs, &bufs);

    // Set whether the second direction is used
    m_trans_dir[R2L] = bidirectional;

    // Push is faster than pull on P8+GPU.
    // Step 1: Reduce-scatter
    // Step 2: Allgather
    for (int step = 0; step < 2; ++step) {    
      for (int i = 0; i < num_total_gpus - 1; ++i) {
      //for (int i = 0; i < 1; ++i) {
        MPI_Request requests[4];
        int num_requests = 0;
        transfer(bufs, work_bufs[L2R], pe_counts, pe_offsets, *streams,
                 i, requests, num_requests, L2R, step == 0);
        if (m_trans_dir[R2L]) {
          int bh_num_requests = 0;
          transfer(bufs, work_bufs[R2L], pe_counts, pe_offsets, *streams,
                   i, requests + num_requests,
                   bh_num_requests, R2L, step == 0);
          num_requests += bh_num_requests;
        }
        ensure_transfer(requests, num_requests, *streams);
        if (step == 0) {
          for (int trans = 0; trans < 2 && m_trans_dir[trans]; ++trans) {
            int devid = trans == L2R ? 0 : m_num_gpus - 1;
            T *dst_base = bufs[devid];
            COLL_CHECK_CUDA(cudaSetDevice(m_gpus[devid]));
            reduce1(dst_base + pe_offsets[trans][m_recv_idx[trans][devid]],
                    work_bufs[trans][devid],
                    pe_counts[trans][m_recv_idx[trans][devid]],
                    streams->at(devid));
            COLL_CHECK_CUDA(cudaEventRecord(m_ev_comp[trans][devid],
                                            streams->at(devid)));
            // notify prev rank for the recording of comp event if MPI
            // is not used
            bool transfer_done_by_cuda =
                m_access_type[trans == L2R ? LHS : RHS] != MPI;
            if (transfer_done_by_cuda) notify_prev_rank(trans);
          }
        } else {
          // Record the event so that the send op can wait for the
          // completion of transfer
          for (int trans = 0; trans < 2 && m_trans_dir[trans]; ++trans) {
            int devid = trans == L2R ? 0 : m_num_gpus - 1;
            COLL_CHECK_CUDA(cudaSetDevice(m_gpus[devid]));
            COLL_CHECK_CUDA(cudaEventRecord(m_ev_comp[trans][devid],
                                            streams->at(devid)));
          }
        }
        for (int g = 0; g < m_num_gpus; ++g) {
          m_send_idx[L2R][g] = dec(m_send_idx[L2R][g], num_total_gpus);
          m_recv_idx[L2R][g] = dec(m_recv_idx[L2R][g], num_total_gpus);
          m_send_idx[R2L][g] = inc(m_send_idx[R2L][g], num_total_gpus);
          m_recv_idx[R2L][g] = inc(m_recv_idx[R2L][g], num_total_gpus);
        }
      }
      if (step == 0) {
        // There remains an un-received notificaton from RHS. 
        if (m_access_type[RHS] != MPI) {
          wait_for_next_rank(L2R);
        }
        if (m_trans_dir[R2L] && m_access_type[LHS] != MPI) {
          wait_for_next_rank(R2L);
        }
      }
#ifdef ALUMINUM_MPI_CUDA_DEBUG
      MPIPrintStream(std::cerr, m_pid)() << "Step " << step << " done\n";
#endif
    }

    // Restore indices
    for (int i = 0; i < 2; ++i) {
      m_send_idx[i] = send_idx_ref[i];
      m_recv_idx[i] = recv_idx_ref[i];
    }
    
    if (!stream_passed) destroy_streams(*streams, m_gpus);

    m_trans_dir[R2L] = true;
                                        
    return 0;
  }

  AccessDir get_src_dir(TransferDir trans_dir) {
    return trans_dir == L2R ? LHS : RHS;
  }
  AccessDir get_dst_dir(TransferDir trans_dir) {
    return trans_dir == L2R ? RHS : LHS;
  }
  
  std::vector<int> m_gpus;
  int m_num_gpus;
  
  MPI_Comm m_comm;  
  std::vector<void*> m_gpu_bufs[2];
  size_t m_gpu_buf_size = 0;

  int m_np;
  int m_pid;
  int m_rid;
  int m_rid_lhs;
  int m_rid_rhs;
  std::vector<int> m_gpu_rid;
  std::vector<int> m_send_idx[2];
  std::vector<int> m_recv_idx[2];

  std::vector<cudaEvent_t> m_ev_comp[2]; // [L2R/R2L]
  std::vector<cudaEvent_t> m_ev_trans[2]; // [L2R/R2L]

  AccessType m_access_type[2];
  int m_neighbor_dev[2];
  void *m_ipc_cache_buf = nullptr;
  void *m_neighbor_buf[2] = {nullptr, nullptr};
  void *m_neighbor_work[2] = {nullptr, nullptr};
  cudaEvent_t m_neighbor_ev[2][2]; // [L2R/R2L][LHS/RHS]

  bool m_trans_dir[2] = {true, true};

  int m_notification_tags[2] = {1001, 1002};
};

} // namespace mpi_cuda
} // namespace internal
} // namespace allreduces
