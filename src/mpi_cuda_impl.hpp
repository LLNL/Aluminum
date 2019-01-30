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
#include "mpi_cuda/allreduce.hpp"

#ifdef AL_HAS_MPI_CUDA_RMA
#include "mpi_cuda/rma.hpp"
#endif

namespace Al {

enum class MPICUDAAllreduceAlgorithm {
  automatic,
  ring,
  bi_ring
};

enum class MPICUDACollectiveAlgorithm {
  automatic
};

inline std::string algorithm_name(MPICUDAAllreduceAlgorithm algo) {
  switch (algo) {
  case MPICUDAAllreduceAlgorithm::automatic:
    return "automatic";
  case MPICUDAAllreduceAlgorithm::ring:
    return "ring";
  case MPICUDAAllreduceAlgorithm::bi_ring:
    return "bi-ring";
  default:
    return "unknown";
  }
}

inline std::string algorithm_name(MPICUDACollectiveAlgorithm algo) {
  switch (algo) {
  case MPICUDACollectiveAlgorithm::automatic:
    return "automatic";
  default:
    return "unknown";
  }
}

namespace internal {
namespace mpi_cuda {

/** Initialize MPI-CUDA backend. */
void init(int& argc, char**& argv);
/** Finalize MPI-CUDA backend. */
void finalize();

/** Represents a request for the MPI-CUDA backend. */
struct MPICUDARequest {
  MPICUDARequest(cudaEvent_t op_event_, cudaStream_t orig_stream_) :
    op_event(op_event_), orig_stream(orig_stream_) {}
  // Note: Not thread safe!
  ~MPICUDARequest() { cuda::release_cuda_event(op_event); }
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
  using allreduce_algo_type = MPICUDAAllreduceAlgorithm;
  using comm_type = internal::mpi_cuda::MPICUDACommunicator;
  using req_type = std::shared_ptr<internal::mpi_cuda::MPICUDARequest>;
  static constexpr std::nullptr_t null_req = nullptr;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type algo) {
    if (count == 0) return;
    switch (algo) {
    case MPICUDAAllreduceAlgorithm::ring:
      internal::mpi_cuda::ring_allreduce(sendbuf, recvbuf, count,
                                         op, comm, comm.get_stream());
      break;
    case MPICUDAAllreduceAlgorithm::automatic:
    case MPICUDAAllreduceAlgorithm::bi_ring:
      internal::mpi_cuda::bi_ring_allreduce(sendbuf, recvbuf, count,
                                            op, comm, comm.get_stream());
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }
  template <typename T>
  static void Allreduce(T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type algo) {
    Allreduce(recvbuf, recvbuf, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingAllreduce(
      const T* sendbuf, T* recvbuf, size_t count,
      ReductionOperator op,
      comm_type& comm,
      req_type& req,
      allreduce_algo_type algo) {
    if (count == 0) return;
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    switch (algo) {
    case MPICUDAAllreduceAlgorithm::ring:
      internal::mpi_cuda::ring_allreduce(sendbuf, recvbuf, count,
                                         op, comm, internal_stream);
      break;
    case MPICUDAAllreduceAlgorithm::automatic:
    case MPICUDAAllreduceAlgorithm::bi_ring:
      internal::mpi_cuda::bi_ring_allreduce(sendbuf, recvbuf, count,
                                            op, comm, internal_stream);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingAllreduce(
      T* recvbuf, size_t count,
      ReductionOperator op, comm_type& comm,
      req_type& req,
      allreduce_algo_type algo) {
    NonblockingAllreduce(recvbuf, recvbuf, count, op, comm,
                         req, algo);
  }

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

 private:
  /** Event for synchronizing between streams. */
  static cudaEvent_t sync_event;
  /**
   * Set up stream synchronization.
   * This will cause the provided internal stream to synchronize with the stream
   * associated with comm.
   */
  static void sync_internal_stream_with_comm(
    cudaStream_t internal_stream, comm_type& comm) {
    // We can reuse a single event for cudaStreamWaitEvent because it uses the
    // stream/event state when it is called.
    AL_CHECK_CUDA(cudaEventRecord(sync_event, comm.get_stream()));
    AL_CHECK_CUDA(cudaStreamWaitEvent(internal_stream, sync_event, 0));
  }
  /**
   * Set up the request for completion checking.
   * This uses an event recorded on the provided internal stream.
   */
  static void setup_completion_event(
    cudaStream_t internal_stream, comm_type& comm, req_type& req) {
    cudaEvent_t event = internal::cuda::get_cuda_event();
    AL_CHECK_CUDA(cudaEventRecord(event, internal_stream));
    req = std::make_shared<internal::mpi_cuda::MPICUDARequest>(event, comm.get_stream());
  }

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
