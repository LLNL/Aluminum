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

#include "cuda_aware_mpi/communicator.hpp"
#include "cuda_aware_mpi/base_state.hpp"

namespace Al {
namespace internal {
namespace cuda_aware_mpi {

/** Progress engine state for CUDA-aware MPI reduce. */
template <typename T>
class ReduceState : public CUDAAwareMPIState {
 public:
  ReduceState(const T* sendbuf_, T* recvbuf_, size_t count_,
              ReductionOperator op_, int root_,
              CUDAAwareMPICommunicator& comm_,
              cudaStream_t stream) :
    CUDAAwareMPIState(comm_, stream),
    comm(comm_),
    sendbuf(sendbuf_), recvbuf(recvbuf_), count(count_),
    op(mpi::ReductionOperator2MPI_Op(op_)), root(root_) {}

  std::string get_name() const override { return "CUDAAwareReduce"; }
  std::string get_desc() const override {
    return "";
  }

 protected:
  void start_mpi_op() override {
    // Data is passed in recvbuf on non-root processes when in-place.
    if (sendbuf == IN_PLACE<T>() && comm.rank() != root) {
      sendbuf = recvbuf;
    }
    MPI_Ireduce(mpi::buf_or_inplace(sendbuf), recvbuf, count,
                mpi::TypeMap<T>(), op, root,
                comm.get_comm(), get_mpi_req());
  }

 private:
  CUDAAwareMPICommunicator& comm;
  const T* sendbuf;
  T* recvbuf;
  size_t count;
  MPI_Op op;
  int root;
};

}  // namespace cuda_aware_mpi
}  // namespace internal
}  // namespace Al
