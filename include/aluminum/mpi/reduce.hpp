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

// Data is passed on recvbuf on non-root processes when in-place.
template <typename T>
void passthrough_reduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, int root, MPICommunicator& comm) {
  if (sendbuf == IN_PLACE<T>() && comm.rank() != root) {
    sendbuf = recvbuf;
  }
  MPI_Reduce(buf_or_inplace(sendbuf), recvbuf, count, TypeMap<T>(),
             ReductionOperator2MPI_Op(op), root, comm.get_comm());
}

template <typename T>
class ReduceAlState : public MPIState {
public:
  ReduceAlState(const T* sendbuf_, T* recvbuf_, size_t count_, ReductionOperator op_,
                int root_, MPICommunicator& comm_, AlRequest req_) :
    MPIState(req_),
    sendbuf(sendbuf_), recvbuf(recvbuf_), count(count_),
    op(ReductionOperator2MPI_Op(op_)), root(root_),
    comm(comm_.get_comm()), rank(comm_.rank()) {}

  ~ReduceAlState() override {}

  std::string get_name() const override { return "MPIReduce"; }

protected:
  void start_mpi_op() override {
    if (sendbuf == IN_PLACE<T>() && rank != root) {
      sendbuf = recvbuf;
    }
    MPI_Ireduce(buf_or_inplace(sendbuf), recvbuf, count, TypeMap<T>(),
                op, root, comm, get_mpi_req());
  }

private:
  const T* sendbuf;
  T* recvbuf;
  size_t count;
  MPI_Op op;
  int root;
  MPI_Comm comm;
  int rank;
};

// Data is passed in recvbuf on non-root processes when in-place.
template <typename T>
void passthrough_nb_reduce(const T* sendbuf, T* recvbuf, size_t count,
                           ReductionOperator op, int root,
                           MPICommunicator& comm, AlRequest& req) {
  req = internal::get_free_request();
  internal::mpi::ReduceAlState<T>* state =
    new internal::mpi::ReduceAlState<T>(
      sendbuf, recvbuf, count, op, root, comm, req);
  get_progress_engine()->enqueue(state);
}

}  // namespace mpi
}  // namespace internal
}  // namespace Al
