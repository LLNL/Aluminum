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

#include <Al_config.hpp>
#include "aluminum/nccl_impl.hpp"
#include "aluminum/mpi/communicator.hpp"

#include <exception>
#include <mutex>
#include <unordered_map>

namespace Al {

// Initialize this.
AlGpuEvent_t NCCLBackend::sync_event = (AlGpuEvent_t) 0;

NCCLCommunicator::NCCLCommunicator() :
  NCCLCommunicator(internal::mpi::get_world_comm().get_comm(), 0) {}

NCCLCommunicator::NCCLCommunicator(MPI_Comm comm_, AlGpuStream_t stream_) :
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

NCCLCommunicator::~NCCLCommunicator() {
  if (m_nccl_comm == nullptr) {
    terminate_al("Attempting to destruct a null NCCLCommunicator");
  }

  int d;
  // Only destroy resources if the driver is still loaded.
  if (AlGpuGetDevice(&d) == AlGpuSuccess) {
    try {
      AL_CHECK_NCCL(ncclCommFinalize(m_nccl_comm));
      AL_CHECK_NCCL(ncclCommDestroy(m_nccl_comm));
    } catch (const al_exception& e) {
      terminate_al("Caught exception in NCCLCommunicator destructor: ",
                   e.what());
    }
    m_nccl_comm = nullptr;
  }
}

namespace internal {
namespace nccl {

void init(int&, char**&) {
  AL_CHECK_CUDA(AlGpuEventCreateWithFlags(&NCCLBackend::sync_event,
                                          AlGpuNoTimingEventFlags));
}

void finalize() {
  AL_CHECK_CUDA(AlGpuEventDestroy(NCCLBackend::sync_event));
}

#ifdef AL_HAS_NCCL_USER_BUFFER_REGISTATION
namespace {
#ifdef AL_THREAD_MULTIPLE
std::mutex nccl_registration_cache_lock;  // Protects access to the map.
#endif
std::unordered_map<void*, void*> nccl_registration_handles;
}
#endif

void register_memory(void* buf, size_t size, NCCLCommunicator& comm) {
#ifdef AL_HAS_NCCL_USER_BUFFER_REGISTATION
  void* handle;
  AL_CHECK_NCCL(ncclCommRegister(comm.get_nccl_comm(), buf, size, &handle));
#ifdef AL_THREAD_MULTIPLE
  std::lock_guard<std::mutex> guard(nccl_registration_cache_lock);
#endif
  nccl_registration_handles[buf] = handle;
#else  // AL_HAS_NCCL_USER_BUFFER_REGISTATION
  (void) buf;
  (void) size;
  (void) comm;
#endif  // AL_HAS_NCCL_USER_BUFFER_REGISTATION
}

void unregister_memory(void* buf, NCCLCommunicator& comm) {
#ifdef AL_HAS_NCCL_USER_BUFFER_REGISTATION
  void* handle;
  {
#ifdef AL_THREAD_MULTIPLE
    std::lock_guard<std::mutex> guard(nccl_registration_cache_lock);
#endif
#ifdef AL_DEBUG
    if (nccl_registration_handles.count(buf) != 1) {
      throw_al_exception(
        "Attempt to unregister memory that was not registered");
    }
#endif
    handle = nccl_registration_handles[buf];
  }
  AL_CHECK_NCCL(ncclCommDeregister(comm.get_nccl_comm(), handle));
#else  // AL_HAS_NCCL_USER_BUFFER_REGISTATION
  (void) buf;
  (void) comm;
#endif  // AL_HAS_NCCL_USER_BUFFER_REGISTATION
}

}  // namespace nccl
}  // namespace internal
}  // namespace Al
