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
#include "aluminum/cuda/gpu_status_flag.hpp"
#include "aluminum/cuda/gpu_wait.hpp"
#include "aluminum/ht/communicator.hpp"

namespace Al {
namespace internal {
namespace ht {

constexpr int pt2pt_tag = 2;

/** GPU point-to-point send operation. */
template <typename T>
class SendAlState : public AlState {
 public:
  SendAlState(const T* sendbuf, size_t count_, int dest_,
              HostTransferCommunicator& comm_, cudaStream_t stream) :
    AlState(nullptr), count(count_), dest(dest_), comm(comm_.get_comm()),
    compute_stream(comm_.get_stream()) {
    mem = mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(count);
    AL_CHECK_CUDA(cudaMemcpyAsync(mem, sendbuf, sizeof(T)*count,
                                  cudaMemcpyDeviceToHost, stream));
    sync_event.record(stream);
  }
  ~SendAlState() override {
    mempool.release<MemoryType::CUDA_PINNED_HOST>(mem);
  }
#ifdef AL_HAS_PROF
  void start() override {
    AlState::start();
    prof_range = profiling::prof_start("HTSend d2h");
  }
#endif
  PEAction step() override {
    if (!mem_transfer_done) {
      if (sync_event.query()) {
        mem_transfer_done = true;
#ifdef AL_HAS_PROF
        profiling::prof_end(prof_range);
#endif
        return PEAction::advance;
      }
      return PEAction::cont;
    }
    if (!send_started) {
#ifdef AL_HAS_PROF
      prof_range = profiling::prof_start("HTSend MPI");
#endif
      MPI_Isend(mem, count, mpi::TypeMap<T>(), dest, pt2pt_tag, comm, &req);
      send_started = true;
    }
    int flag;
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
#ifdef AL_HAS_PROF
    if (flag) {
      profiling::prof_end(prof_range);
    }
#endif
    return flag ? PEAction::complete : PEAction::cont;
  }
  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }
  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "HTSend"; }
  std::string get_desc() const override {
    return std::to_string(count) + " " + std::to_string(dest);
  }
 private:
  T* mem;
  size_t count;
  int dest;
  MPI_Comm comm;
  MPI_Request req = MPI_REQUEST_NULL;
  cuda::GPUStatusFlag sync_event;
  bool mem_transfer_done = false;
  bool send_started = false;
  cudaStream_t compute_stream;
#ifdef AL_HAS_PROF
  profiling::ProfileRange prof_range;
#endif
};

template <typename T>
class RecvAlState : public AlState {
 public:
  RecvAlState(T* recvbuf, size_t count_, int src_,
              HostTransferCommunicator& comm_, cudaStream_t stream) :
    AlState(nullptr), count(count_), src(src_), comm(comm_.get_comm()),
    compute_stream(comm_.get_stream()) {
    mem = mempool.allocate<MemoryType::CUDA_PINNED_HOST, T>(count);
    wait_sync.wait(stream);
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
    sync_event.record(stream);
  }
  ~RecvAlState() override {
    mempool.release<MemoryType::CUDA_PINNED_HOST>(mem);
  }
  void start() override {
    AlState::start();
#ifdef AL_HAS_PROF
    prof_range = profiling::prof_start("HTRecv MPI");
#endif
    MPI_Irecv(mem, count, mpi::TypeMap<T>(), src, pt2pt_tag, comm, &req);
  }
  PEAction step() override {
    if (!recv_done) {
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        recv_done = true;
        // Signal the device that the memcpy can start.
        wait_sync.signal();
#ifdef AL_HAS_PROF
        profiling::prof_end(prof_range);
        prof_range = profiling::prof_start("HTRecv h2d");
#endif
      }
    }
    // Wait until the memcpy has completed so everything can be safely freed.
#ifdef AL_HAS_PROF
    bool r = sync_event.query();
    if (r) {
      profiling::prof_end(prof_range);
    }
    return r ? PEAction::complete : PEAction::cont;
#else
    return sync_event.query() ? PEAction::complete : PEAction::cont;
#endif
  }
  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }
  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "HTRecv"; }
  std::string get_desc() const override {
    return std::to_string(count) + " " + std::to_string(src);
  }
 private:
  T* mem;
  size_t count;
  int src;
  MPI_Comm comm;
  MPI_Request req = MPI_REQUEST_NULL;
  cuda::GPUStatusFlag sync_event;
  cuda::GPUWait wait_sync;
  bool recv_done = false;
  cudaStream_t compute_stream;
#ifdef AL_HAS_PROF
  profiling::ProfileRange prof_range;
#endif
};

template <typename T>
class SendRecvAlState : public AlState {
 public:
  SendRecvAlState(const T* sendbuf, size_t send_count, int dest,
                  T* recvbuf, size_t recv_count, int src,
                  HostTransferCommunicator& comm, cudaStream_t stream) :
    AlState(nullptr),
    send_state(sendbuf, send_count, dest, comm, stream),
    recv_state(recvbuf, recv_count, src, comm, stream) {}
  void start() override {
    AlState::start();
    send_state.start();
    recv_state.start();
  }
  PEAction step() override {
    if (!send_done) {
      PEAction send_action = send_state.step();
      if (send_action == PEAction::advance) {
        return send_action;
      }
      send_done = send_action == PEAction::complete;
    }
    if (!recv_done) {
      recv_done = recv_state.step() == PEAction::complete;
    }
    return send_done && recv_done ? PEAction::complete : PEAction::cont;
  }
  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override {
    return send_state.get_compute_stream();
  }
  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "HTSendRecv"; }
  std::string get_desc() const override {
    return send_state.get_desc() + " " + recv_state.get_desc();
  }
 private:
  SendAlState<T> send_state;
  RecvAlState<T> recv_state;
  bool send_advanced = false;
  bool send_done = false;
  bool recv_done = false;
};


}  // namespace ht
}  // namespace internal
}  // namespace Al
