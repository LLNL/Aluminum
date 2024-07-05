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

#include "aluminum/progress.hpp"
#include "aluminum/mpi/base_state.hpp"
#include "aluminum/mpi/communicator.hpp"
#include "aluminum/mpi/utils.hpp"
#include "aluminum/utils/caching_allocator.hpp"
#include "aluminum/mempool.hpp"
#include <algorithm>
#include <mpi.h>

namespace Al {
namespace internal {
namespace mpi {

template <typename T>
void passthrough_send(const T* sendbuf, size_t count, int dest,
                      MPICommunicator& comm) {
  AL_MPI_LARGE_COUNT_CALL(MPI_Send)(
    sendbuf, count, TypeMap<T>(), dest, pt2pt_tag, comm.get_comm());
}

/** GPU point-to-point send operation. */
template <typename T>
class SendAlState : public MPIState {
 public:
  SendAlState(const T* sendbuf_, size_t count_, int dest_,
              MPICommunicator& comm_, AlMPIReq req_) :
    MPIState(req_),
    sendbuf(sendbuf_), count(count_), dest(dest_),
    comm(comm_.get_comm()) {}

  ~SendAlState() override {}

  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "MPISend"; }

protected:
  void start_mpi_op() override {
    AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
      sendbuf, count, TypeMap<T>(), dest, pt2pt_tag, comm, get_mpi_req());
  }

 private:
  const T* sendbuf;
  size_t count;
  int dest;
  MPI_Comm comm;
};

template <typename T>
void passthrough_nb_send(const T* sendbuf, size_t count, int dest,
                         MPICommunicator& comm, AlMPIReq& req) {
  req = get_free_request();
  internal::mpi::SendAlState<T>* state =
    new internal::mpi::SendAlState<T>(
      sendbuf, count, dest, comm, req);
  get_progress_engine()->enqueue(state);
}

template <typename T>
void passthrough_recv(T* recvbuf, size_t count, int src,
                      MPICommunicator& comm) {
  AL_MPI_LARGE_COUNT_CALL(MPI_Recv)(
    recvbuf, count, TypeMap<T>(), src, pt2pt_tag, comm.get_comm(),
    MPI_STATUS_IGNORE);
}

template <typename T>
class RecvAlState : public MPIState {
 public:
  RecvAlState(T* recvbuf_, size_t count_, int src_,
              MPICommunicator& comm_, AlMPIReq req_) :
    MPIState(req_),
    recvbuf(recvbuf_), count(count_), src(src_),
    comm(comm_.get_comm()) {}

  ~RecvAlState() override {}

  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "MPIRecv"; }

protected:
  void start_mpi_op() override {
    AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
      recvbuf, count, mpi::TypeMap<T>(), src, pt2pt_tag, comm, get_mpi_req());
  }

 private:
  T* recvbuf;
  size_t count;
  int src;
  MPI_Comm comm;
};

template <typename T>
void passthrough_nb_recv(T* recvbuf, size_t count, int src,
                         MPICommunicator& comm, AlMPIReq& req) {
  req = get_free_request();
  internal::mpi::RecvAlState<T>* state =
    new internal::mpi::RecvAlState<T>(
      recvbuf, count, src, comm, req);
  get_progress_engine()->enqueue(state);
}

template <typename T>
void passthrough_sendrecv(const T* sendbuf, size_t send_count, int dest,
                          T* recvbuf, size_t recv_count, int src,
                          MPICommunicator& comm) {
  if (sendbuf == internal::IN_PLACE<T>()) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Sendrecv_replace)(
      recvbuf, recv_count, TypeMap<T>(), dest, pt2pt_tag,
      src, pt2pt_tag, comm.get_comm(), MPI_STATUS_IGNORE);
  } else {
    AL_MPI_LARGE_COUNT_CALL(MPI_Sendrecv)(
      sendbuf, send_count, TypeMap<T>(), dest, pt2pt_tag,
      recvbuf, recv_count, TypeMap<T>(), src, pt2pt_tag,
      comm.get_comm(), MPI_STATUS_IGNORE);
  }
}

template <typename T>
class SendRecvAlState : public MPIState {
 public:
  SendRecvAlState(const T* sendbuf_, size_t send_count_, int dest_,
                  T* recvbuf_, size_t recv_count_, int src_,
                  MPICommunicator& comm_, AlMPIReq req_) :
    MPIState(req_),
    sendbuf(sendbuf_), send_count(send_count_), dest(dest_),
    recvbuf(recvbuf_), recv_count(recv_count_), src(src_),
    comm(comm_.get_comm()), tmp_buf(nullptr) {
    if (sendbuf == internal::IN_PLACE<T>()) {
      tmp_buf = internal::mempool.allocate<internal::MemoryType::HOST, T>(
        recv_count);
    }
  }

  ~SendRecvAlState() {
    if (tmp_buf) {
      internal::mempool.release<internal::MemoryType::HOST>(tmp_buf);
      tmp_buf = nullptr;
    }
  }

  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "MPISendRecv"; }

protected:
  void start_mpi_op() override {
    // Note: MPI_Isendrecv(_replace) was added in MPI 4.0,
    // which is probably too new.
    if (sendbuf == internal::IN_PLACE<T>()) {
      // Copy the send buffer to the temporary buffer.
      std::copy_n(recvbuf, recv_count, tmp_buf);
      AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
        recvbuf, recv_count, TypeMap<T>(), src, pt2pt_tag, comm,
        &mpi_reqs[0]);
      AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
        tmp_buf, recv_count, TypeMap<T>(), dest, pt2pt_tag, comm,
        &mpi_reqs[1]);
    } else {
      AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
        recvbuf, recv_count, TypeMap<T>(), src, pt2pt_tag, comm,
        &mpi_reqs[0]);
      AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
        sendbuf, send_count, TypeMap<T>(), dest, pt2pt_tag, comm,
        &mpi_reqs[1]);
    }
  }

  bool poll_mpi() override {
    int flag;
    MPI_Testall(2, mpi_reqs, &flag, MPI_STATUSES_IGNORE);
    return flag;
  }

 private:
  const T* sendbuf;
  size_t send_count;
  int dest;
  T* recvbuf;
  size_t recv_count;
  int src;
  MPI_Comm comm;
  MPI_Request mpi_reqs[2];
  T* tmp_buf;
};

template <typename T>
void passthrough_nb_sendrecv(const T* sendbuf, size_t send_count, int dest,
                             T* recvbuf, size_t recv_count, int src,
                             MPICommunicator& comm, AlMPIReq& req) {
  req = get_free_request();
  internal::mpi::SendRecvAlState<T>* state =
    new internal::mpi::SendRecvAlState<T>(
      sendbuf, send_count, dest, recvbuf, recv_count, src, comm, req);
  get_progress_engine()->enqueue(state);
}


}  // namespace mpi
}  // namespace internal
}  // namespace Al
