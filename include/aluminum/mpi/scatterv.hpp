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

// Data is passed in recvbuf on root processes when in-place.
template <typename T>
void passthrough_scatterv(const T* sendbuf, T* recvbuf,
                          std::vector<size_t> counts,
                          std::vector<size_t> displs,
                          int root,
                          MPICommunicator& comm) {
  auto counts_ = countify_size_t_vector(counts);
  auto displs_ = displify_size_t_vector(displs);
  if (sendbuf == IN_PLACE<T>() && comm.rank() == root) {
    sendbuf = recvbuf;
    recvbuf = IN_PLACE<T>();
  }
  AL_MPI_LARGE_COUNT_CALL(MPI_Scatterv)(
    sendbuf, counts_.data(), displs_.data(), TypeMap<T>(),
    buf_or_inplace(recvbuf), counts[comm.rank()], TypeMap<T>(),
    root, comm.get_comm());
}

template <typename T>
class ScattervAlState : public MPIState {
public:
  ScattervAlState(const T* sendbuf_, T* recvbuf_,
                  std::vector<size_t> counts_,
                  std::vector<size_t> displs_,
                  int root_,
                  MPICommunicator& comm_, AlMPIReq req_) :
    MPIState(req_),
    sendbuf(sendbuf_), recvbuf(recvbuf_),
    counts(countify_size_t_vector(counts_)),
    displs(displify_size_t_vector(displs_)),
    root(root_),
    comm(comm_.get_comm()), rank(comm_.rank()) {}

  ~ScattervAlState() override {}

  std::string get_name() const override { return "MPIScatterv"; }

protected:
  void start_mpi_op() override {
    if (sendbuf == IN_PLACE<T>() && rank == root) {
      sendbuf = recvbuf;
      recvbuf = IN_PLACE<T>();
    }
    AL_MPI_LARGE_COUNT_CALL(MPI_Iscatterv)(
      sendbuf, counts.data(), displs.data(), TypeMap<T>(),
      buf_or_inplace(recvbuf), counts[rank], TypeMap<T>(),
      root, comm, get_mpi_req());
  }

private:
  const T* sendbuf;
  T* recvbuf;
  Al_mpi_count_vector_t counts;
  Al_mpi_displ_vector_t displs;
  int root;
  MPI_Comm comm;
  int rank;
};

// When in-place, it is recvbuf that uses IN_PLACE.
template <typename T>
void passthrough_nb_scatterv(const T* sendbuf, T* recvbuf,
                             std::vector<size_t> counts,
                             std::vector<size_t> displs,
                             int root, MPICommunicator& comm, AlMPIReq& req) {
  req = get_free_request();
  internal::mpi::ScattervAlState<T>* state =
    new internal::mpi::ScattervAlState<T>(
      sendbuf, recvbuf, counts, displs, root, comm, req);
  get_progress_engine()->enqueue(state);
}

}  // namespace mpi
}  // namespace internal
}  // namespace Al
