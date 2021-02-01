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

namespace Al {
namespace internal {
namespace mpi {

template <typename T>
void passthrough_alltoallv(const T* sendbuf,
                           std::vector<size_t> send_counts,
                           std::vector<size_t> send_displs,
                           T* recvbuf,
                           std::vector<size_t> recv_counts,
                           std::vector<size_t> recv_displs,
                           MPICommunicator& comm) {
  std::vector<int> send_counts_ = intify_size_t_vector(send_counts);
  std::vector<int> send_displs_ = intify_size_t_vector(send_displs);
  std::vector<int> recv_counts_ = intify_size_t_vector(recv_counts);
  std::vector<int> recv_displs_ = intify_size_t_vector(recv_displs);
  MPI_Alltoallv(buf_or_inplace(sendbuf),
                send_counts_.data(), send_displs_.data(), TypeMap<T>(),
                recvbuf,
                recv_counts_.data(), recv_displs_.data(), TypeMap<T>(),
                comm.get_comm());
}

template <typename T>
class AlltoallvAlState : public MPIState {
public:
  AlltoallvAlState(const T* sendbuf_,
                   std::vector<size_t> send_counts_,
                   std::vector<size_t> send_displs_,
                   T* recvbuf_,
                   std::vector<size_t> recv_counts_,
                   std::vector<size_t> recv_displs_,
                   MPICommunicator& comm_, AlRequest req_) :
    MPIState(req_),
    sendbuf(sendbuf_),
    send_counts(intify_size_t_vector(send_counts_)),
    send_displs(intify_size_t_vector(send_displs_)),
    recvbuf(recvbuf_),
    recv_counts(intify_size_t_vector(recv_counts_)),
    recv_displs(intify_size_t_vector(recv_displs_)),
    comm(comm_.get_comm()) {}

  ~AlltoallvAlState() override {}

  std::string get_name() const override { return "MPIAlltoallv"; }

protected:
  void start_mpi_op() override {
    MPI_Ialltoallv(buf_or_inplace(sendbuf),
                   send_counts.data(), send_displs.data(), TypeMap<T>(),
                   recvbuf,
                   recv_counts.data(), recv_displs.data(), TypeMap<T>(),
                   comm, get_mpi_req());
  }

private:
  const T* sendbuf;
  std::vector<int> send_counts;
  std::vector<int> send_displs;
  T* recvbuf;
  std::vector<int> recv_counts;
  std::vector<int> recv_displs;
  MPI_Comm comm;
};

template <typename T>
void passthrough_nb_alltoallv(const T* sendbuf,
                              std::vector<size_t> send_counts,
                              std::vector<size_t> send_displs,
                              T* recvbuf,
                              std::vector<size_t> recv_counts,
                              std::vector<size_t> recv_displs,
                              MPICommunicator& comm, AlRequest& req) {
  req = internal::get_free_request();
  internal::mpi::AlltoallvAlState<T>* state =
    new internal::mpi::AlltoallvAlState<T>(
      sendbuf, send_counts, send_displs,
      recvbuf, recv_counts, recv_displs, comm, req);
  get_progress_engine()->enqueue(state);
}

}  // namespace mpi
}  // namespace internal
}  // namespace Al
