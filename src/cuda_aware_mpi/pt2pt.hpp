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

#include "cuda.hpp"
#include "progress.hpp"
#include "cuda_aware_mpi_impl.hpp"
#include "mpi_impl.hpp"
#include "cuda_aware_mpi/communicator.hpp"
#include "cuda_aware_mpi/base_state.hpp"

namespace Al {
namespace internal {
namespace cuda_aware_mpi {

constexpr int pt2pt_tag = 2;

/** Progress engine state for CUDA-aware MPI send. */
template <typename T>
class SendState : public CUDAAwareMPIState {
 public:
  SendState(const T* sendbuf_, size_t count_, int dest_,
            CUDAAwareMPICommunicator& comm_,
            cudaStream_t stream) :
    CUDAAwareMPIState(stream),
    compute_stream(comm_.get_stream()),
    comm(comm_),
    sendbuf(sendbuf_), count(count_), dest(dest_) {}

  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }
  RunType get_run_type() const override { return RunType::unbounded; }

  std::string get_name() const override { return "CUDAAwareSend"; }
  std::string get_desc() const override {
    return "";
  }

 protected:
  void start_mpi_op() override {
    MPI_Isend(sendbuf, count, mpi::TypeMap<T>(), dest,
              pt2pt_tag, comm.get_comm(), get_mpi_req());
  }
 private:
  cudaStream_t compute_stream;

  CUDAAwareMPICommunicator& comm;
  const T* sendbuf;
  size_t count;
  int dest;
};

/** Progress engine state for CUDA-aware MPI recv. */
template <typename T>
class RecvState : public CUDAAwareMPIState {
 public:
  RecvState(T* recvbuf_, size_t count_, int src_,
            CUDAAwareMPICommunicator& comm_,
            cudaStream_t stream) :
    CUDAAwareMPIState(stream),
    compute_stream(comm_.get_stream()),
    comm(comm_),
    recvbuf(recvbuf_), count(count_), src(src_) {}

  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }
  RunType get_run_type() const override { return RunType::unbounded; }

  std::string get_name() const override { return "CUDAAwareRecv"; }
  std::string get_desc() const override {
    return "";
  }

 protected:
  void start_mpi_op() override {
    MPI_Irecv(recvbuf, count, mpi::TypeMap<T>(), src,
              pt2pt_tag, comm.get_comm(), get_mpi_req());
  }
 private:
  cudaStream_t compute_stream;

  CUDAAwareMPICommunicator& comm;
  T* recvbuf;
  size_t count;
  int src;
};

// Doesn't fit CUDAAwareMPIState pattern due to two requests.
/** Progress engine state for CUDA-aware MPI send/recv. */
template <typename T>
class SendRecvState : public AlState {
 public:
  SendRecvState(const T* sendbuf_, size_t send_count_, int dest_,
                T* recvbuf_, size_t recv_count_, int src_,
                CUDAAwareMPICommunicator& comm_,
                cudaStream_t stream) :
    AlState(nullptr),
    compute_stream(comm_.get_stream()),
    comm(comm_),
    sendbuf(sendbuf_), send_count(send_count_), dest(dest_),
    recvbuf(recvbuf_), recv_count(recv_count_), src(src_) {
    // To check that all prior computation has completed.
    pending_ops_event.record(stream);
    // To ensure nothing modifies the buffer until ready.
    gpu_wait.wait(stream);
  }

  ~SendRecvState() {}

  bool step() override {
    if (!op_ready) {
      // Wait for pending operations.
      if (pending_ops_event.query()) {
        op_ready = true;
      }
      // Always return false here to ensure operations start in the right
      // order.
      return false;
    }
    if (!op_started) {
      MPI_Irecv(recvbuf, recv_count, mpi::TypeMap<T>(), src,
                pt2pt_tag, comm.get_comm(), &mpi_reqs[0]);
      MPI_Isend(sendbuf, send_count, mpi::TypeMap<T>(), dest,
                pt2pt_tag, comm.get_comm(), &mpi_reqs[1]);
      op_started = true;
    }
    int flag;
    MPI_Testall(2, mpi_reqs, &flag, MPI_STATUSES_IGNORE);
    if (flag) {
      gpu_wait.signal();
      return true;
    } else {
      return false;
    }
  }

  bool needs_completion() const override { return false; }
  void* get_compute_stream() const override { return compute_stream; }
  RunType get_run_type() const override { return RunType::unbounded; }

  std::string get_name() const override { return "CUDAAwareSendRecv"; }
  std::string get_desc() const override {
    return "";
  }
 private:
  cuda::FastEvent pending_ops_event;
  cuda::GPUWait gpu_wait;
  cudaStream_t compute_stream;
  bool op_ready = false;
  bool op_started = false;

  CUDAAwareMPICommunicator& comm;
  const T* sendbuf;
  size_t send_count;
  int dest;
  T* recvbuf;
  size_t recv_count;
  int src;
  MPI_Request mpi_reqs[2];
};

}  // namespace cuda_aware_mpi
}  // namespace internal
}  // namespace Al
