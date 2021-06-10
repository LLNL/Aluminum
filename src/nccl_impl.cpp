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

#include "aluminum/nccl_impl.hpp"

namespace Al {

// Initialize this.
cudaEvent_t NCCLBackend::sync_event = (cudaEvent_t) 0;

NCCLCommunicator::NCCLCommunicator(MPI_Comm comm_, cudaStream_t stream_) :
  MPICommAndStreamWrapper(comm_, stream_) {
  // Get a unique ID for this communicator from NCCL and distribute it.
  ncclUniqueId nccl_id;
  if (rank() == 0) {
    AL_CHECK_NCCL(ncclGetUniqueId(&nccl_id));
  }
  MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, get_comm());
  // This uses the current CUDA device.
  AL_CHECK_NCCL(ncclCommInitRank(&m_nccl_comm, size(), nccl_id, rank()));
}

NCCLCommunicator::NCCLCommunicator(int rank_, int size_, ncclUniqueId nccl_id_, cudaStream_t stream_) :
  MPICommAndStreamWrapper(rank_, size_, stream_) {
  // This uses the current CUDA device.
  AL_CHECK_NCCL(ncclCommInitRank(&m_nccl_comm, size_, nccl_id_, rank_));
}

NCCLCommunicator::~NCCLCommunicator() {
  int d;
  // Only destroy resources if the driver is still loaded.
  if (cudaGetDevice(&d) == cudaSuccess) {
    try {
      AL_CHECK_NCCL(ncclCommDestroy(m_nccl_comm));
    } catch (const al_exception& e) {
      std::cerr << "Caught exception in NCCLCommunicator destructor: "
                << e.what() << std::endl;
      std::terminate();
    }
  }
}

namespace internal {
namespace nccl {

void init(int&, char**&) {
  AL_CHECK_CUDA(cudaEventCreateWithFlags(&NCCLBackend::sync_event,
                                         cudaEventDisableTiming));
}

void finalize() {
  AL_CHECK_CUDA(cudaEventDestroy(NCCLBackend::sync_event));
}

}  // namespace nccl
}  // namespace internal
}  // namespace Al
