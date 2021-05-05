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

#include "Al.hpp"
#include "aluminum/mpi_cuda/communicator.hpp"
#include "aluminum/cuda/events.hpp"
#ifdef AL_HAS_MPI_CUDA_RMA
#include "aluminum/mpi_cuda/rma.hpp"
#endif

namespace Al {

namespace internal {
namespace mpi_cuda {

/** Initialize MPI-CUDA backend. */
void init(int& argc, char**& argv);
/** Finalize MPI-CUDA backend. */
void finalize();

// TODO: Not used.
/** Represents a request for the MPI-CUDA backend. */
struct MPICUDARequest {
  MPICUDARequest(cudaEvent_t op_event_, cudaStream_t orig_stream_) :
    op_event(op_event_), orig_stream(orig_stream_) {}
  ~MPICUDARequest() { cuda::event_pool.release(op_event); }
  /** Event pending on completion of the operation. */
  cudaEvent_t op_event;
  /** Original stream associated with the operation. */
  cudaStream_t orig_stream;
};

}  // namespace mpi_cuda
}  // namespace internal

class MPICUDABackend {
  friend void internal::mpi_cuda::init(int&, char**&);
  friend void internal::mpi_cuda::finalize();
 public:
  using comm_type = internal::mpi_cuda::MPICUDACommunicator;
  using req_type = std::shared_ptr<internal::mpi_cuda::MPICUDARequest>;
  static constexpr std::nullptr_t null_req = nullptr;

#ifdef AL_HAS_MPI_CUDA_RMA
  template <typename T>
  static T *AttachRemoteBuffer(T *local_buf, int peer, comm_type& comm) {
    return static_cast<T*>(
        comm.get_rma().attach_remote_buffer(local_buf, peer));
  }

  template <typename T>
  static void DetachRemoteBuffer(T *remote_buf, int peer, comm_type& comm) {
    comm.get_rma().detach_remote_buffer(remote_buf, peer);
  }

  static void Notify(int peer, comm_type& comm) {
    comm.get_rma().notify(peer);
  }

  static void Wait(int peer, comm_type& comm) {
    comm.get_rma().wait(peer);
  }

  static void Sync(int peer, comm_type& comm) {
    comm.get_rma().sync(peer);
  }

  static void Sync(const int *peers, int num_peers, comm_type& comm) {
    comm.get_rma().sync(peers, num_peers);
  }

  template <typename T>
  static void Put(
      const T* srcbuf, int dest, T * destbuf, size_t count,
      comm_type& comm) {
    comm.get_rma().put(srcbuf, dest, destbuf, sizeof(T) * count);
  }
#endif // AL_HAS_MPI_CUDA_RMA

  static std::string Name() { return "MPICUDABackend"; }
};

template <>
inline bool Test<MPICUDABackend>(typename MPICUDABackend::req_type& req) {
  if (req == MPICUDABackend::null_req) {
    return true;
  }
  // This is purely a host operation.
  bool r = cudaEventQuery(req->op_event) == cudaSuccess;
  if (r) {
    req = MPICUDABackend::null_req;
  }
  return r;
}

template <>
inline void Wait<MPICUDABackend>(typename MPICUDABackend::req_type& req) {
  if (req == MPICUDABackend::null_req) {
    return;
  }
  // Synchronize the original stream with the request.
  // This will not block the host.
  AL_CHECK_CUDA(cudaStreamWaitEvent(req->orig_stream, req->op_event, 0));
}

}  // namespace Al
