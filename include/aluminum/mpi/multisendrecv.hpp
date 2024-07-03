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

#include <vector>

#include "aluminum/progress.hpp"
#include "aluminum/mpi/base_state.hpp"
#include "aluminum/mpi/communicator.hpp"
#include "aluminum/mpi/utils.hpp"
#include "aluminum/utils/caching_allocator.hpp"
#include "aluminum/mempool.hpp"
#include "base_state.hpp"
#include <algorithm>
#include <numeric>
#include <mpi.h>

namespace Al {
namespace internal {
namespace mpi {

template <typename T>
void passthrough_multisendrecv(std::vector<const T*> send_buffers,
                               std::vector<size_t> send_counts,
                               std::vector<int> dests,
                               std::vector<T*> recv_buffers,
                               std::vector<size_t> recv_counts,
                               std::vector<int> srcs,
                               MPICommunicator& comm) {
  if (send_buffers.empty() && recv_buffers.empty()) {
    return;
  }
  std::vector<MPI_Request> reqs(send_buffers.size() + recv_buffers.size());
  for (size_t i = 0; i < recv_buffers.size(); ++i) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
      recv_buffers[i], recv_counts[i], TypeMap<T>(), srcs[i],
      pt2pt_tag, comm.get_comm(), &reqs[i]);
  }
  for (size_t i = 0; i < send_buffers.size(); ++i) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
      send_buffers[i], send_counts[i], TypeMap<T>(), dests[i],
      pt2pt_tag, comm.get_comm(), &reqs[recv_buffers.size() + i]);
  }
  MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
}

template <typename T>
void passthrough_inplace_multisendrecv(std::vector<T*> buffers,
                                       std::vector<size_t> counts,
                                       std::vector<int> dests,
                                       std::vector<int> srcs,
                                       MPICommunicator &comm) {
  if (buffers.empty()) {
    return;
  }
  // Allocate a single temporary buffer for sending and split it.
  // Cannot use Isendrecv_replace since MPI 4.0 is probably too new.
  std::vector<const T*> tmp_buffers(buffers.size(), nullptr);
  size_t total_size = std::accumulate(counts.begin(), counts.end(), size_t{0});
  T* tmp_buf = nullptr;
  if (total_size > 0) {
    tmp_buf =
      internal::mempool.allocate<internal::MemoryType::HOST, T>(total_size);
  }
  // Do the copy through the non-const pointer.
  T* cur_ptr = nullptr;
  for (size_t i = 0; i < buffers.size(); ++i) {
    cur_ptr = (i == 0) ? tmp_buf : cur_ptr + counts[i-1];
    std::copy_n(buffers[i], counts[i], cur_ptr);
    tmp_buffers[i] = cur_ptr;
  }
  passthrough_multisendrecv(tmp_buffers, counts, dests, buffers, counts, srcs, comm);
  internal::mempool.release<internal::MemoryType::HOST>(tmp_buf);
}

template <typename T>
class MultiSendRecvAlState : public MPIState {
public:
  MultiSendRecvAlState(std::vector<const T*> send_buffers_,
                       std::vector<size_t> send_counts_,
                       std::vector<int> dests_,
                       std::vector<T*> recv_buffers_,
                       std::vector<size_t> recv_counts_,
                       std::vector<int> srcs_,
                       MPICommunicator& comm_,
                       AlMPIReq req_) :
    MPIState(req_),
    send_buffers(std::move(send_buffers_)),
    send_counts(send_counts_),
    dests(std::move(dests_)),
    recv_buffers(std::move(recv_buffers_)),
    recv_counts(recv_counts_),
    srcs(std::move(srcs_)),
    comm(comm_.get_comm()),
    mpi_reqs(send_buffers.size() + recv_buffers.size())
  {}

  // In-place version which sets up necessary buffers/etc.
  MultiSendRecvAlState(std::vector<T*> buffers,
                       std::vector<size_t> counts,
                       std::vector<int> dests_,
                       std::vector<int> srcs_,
                       MPICommunicator& comm_,
                       AlMPIReq req_) :
    MPIState(req_),
    send_buffers(buffers.size(), nullptr),
    send_counts(counts),
    dests(std::move(dests_)),
    recv_buffers(std::move(buffers)),
    recv_counts(counts),
    srcs(std::move(srcs_)),
    comm(comm_.get_comm()),
    mpi_reqs(send_buffers.size() + recv_buffers.size()) {
    // Allocate space and set up pointers but do not copy.
    size_t total_size = std::accumulate(counts.begin(), counts.end(), size_t{0});
    if (total_size > 0)
    {
      tmp_buf =
        internal::mempool.allocate<internal::MemoryType::HOST, T>(total_size);
    }
    if (!send_buffers.empty()) {
      send_buffers[0] = tmp_buf;
      for (size_t i = 1; i < counts.size(); ++i) {
        send_buffers[i] = send_buffers[i-1] + counts[i-1];
      }
    }
  }

  ~MultiSendRecvAlState() override {
    if (tmp_buf) {
      internal::mempool.release<internal::MemoryType::HOST>(tmp_buf);
      tmp_buf = nullptr;
    }
  }

  RunType get_run_type() const override { return RunType::unbounded; }
  std::string get_name() const override { return "MPIMultiSendRecv"; }

protected:
  void start_mpi_op() override {
    // Copy data if needed.
    if (tmp_buf) {
      T* cur_ptr = nullptr;
      for (size_t i = 0; i < send_buffers.size(); ++i) {
        cur_ptr = (i == 0) ? tmp_buf : cur_ptr + send_counts[i-1];
        std::copy_n(recv_buffers[i], send_counts[i], cur_ptr);
      }
    }
    for (size_t i = 0; i < recv_buffers.size(); ++i) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
        recv_buffers[i], recv_counts[i], TypeMap<T>(), srcs[i],
        pt2pt_tag, comm, &mpi_reqs[i]);
    }
    for (size_t i = 0; i < send_buffers.size(); ++i) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
        send_buffers[i], send_counts[i], TypeMap<T>(), dests[i],
        pt2pt_tag, comm, &mpi_reqs[recv_buffers.size() + i]);
    }
  }

  bool poll_mpi() override {
    int flag;
    MPI_Testall(mpi_reqs.size(), mpi_reqs.data(), &flag, MPI_STATUSES_IGNORE);
    return flag;
  }

private:
  std::vector<const T*> send_buffers;
  std::vector<size_t> send_counts;
  std::vector<int> dests;
  std::vector<T*> recv_buffers;
  std::vector<size_t> recv_counts;
  std::vector<int> srcs;
  MPI_Comm comm;
  std::vector<MPI_Request> mpi_reqs;
  T* tmp_buf = nullptr;
};

template <typename T>
void passthrough_nb_multisendrecv(std::vector<const T*> send_buffers,
                                  std::vector<size_t> send_counts,
                                  std::vector<int> dests,
                                  std::vector<T*> recv_buffers,
                                  std::vector<size_t> recv_counts,
                                  std::vector<int> srcs,
                                  MPICommunicator& comm,
                                  AlMPIReq& req) {
  if (send_buffers.empty() && recv_buffers.empty()) {
    req = nullptr;
  }
  req = get_free_request();
  internal::mpi::MultiSendRecvAlState<T>* state =
    new internal::mpi::MultiSendRecvAlState<T>(
      send_buffers, send_counts, dests, recv_buffers, recv_counts, srcs,
      comm, req);
  get_progress_engine()->enqueue(state);
}

template <typename T>
void passthrough_nb_inplace_multisendrecv(std::vector<T *> buffers,
                                          std::vector<size_t> counts,
                                          std::vector<int> dests,
                                          std::vector<int> srcs,
                                          MPICommunicator &comm,
                                          AlMPIReq &req) {
  if (buffers.empty()) {
    req = nullptr;
  }
  req = get_free_request();
  internal::mpi::MultiSendRecvAlState<T>* state =
    new internal::mpi::MultiSendRecvAlState<T>(
      buffers, counts, dests, srcs, comm, req);
  get_progress_engine()->enqueue(state);
}

}  // namespace mpi
}  // namespace internal
}  // namespace Al
