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

#include "Al_config.hpp"

#if defined(AL_HAS_ROCM)
#include <rccl.h>
#elif defined(AL_HAS_CUDA)
#include <nccl.h>
#endif // defined(AL_HAS_ROCM)

#include "Al.hpp"
#include "internal.hpp"
#include "cuda.hpp"
#include "cudacommunicator.hpp"

#define AL_FORCE_CHECK_NCCL(nccl_call)                                \
  do {                                                                \
    AL_CUDA_SYNC(true);                                               \
    ncclResult_t result_CHECK_NCCL = (nccl_call);                     \
    if (result_CHECK_NCCL != ncclSuccess) {                           \
      throw_al_exception(std::string("NCCL error: ")                  \
                         + ncclGetErrorString(result_CHECK_NCCL));    \
    }                                                                 \
    AL_CUDA_SYNC(false);                                              \
  } while (0)
#define AL_FORCE_CHECK_NCCL_NOSYNC(nccl_call)                         \
  do {                                                                \
    ncclResult_t result_CHECK_NCCL = (nccl_call);                     \
    if (result_CHECK_NCCL != ncclSuccess) {                           \
      throw_al_exception(std::string("NCCL error: ")                  \
                         + ncclGetErrorString(result_CHECK_NCCL));    \
    }                                                                 \
  } while (0)
#ifdef AL_DEBUG
#define AL_CHECK_NCCL(nccl_call) AL_FORCE_CHECK_NCCL_NOSYNC(nccl_call)
#else
#define AL_CHECK_NCCL(nccl_call) AL_FORCE_CHECK_NCCL_NOSYNC(nccl_call)
#endif

namespace Al {

enum class NCCLCollectiveAlgorithm {
  automatic
};

inline std::string algorithm_name(NCCLCollectiveAlgorithm algo) {
  switch (algo) {
  case NCCLCollectiveAlgorithm::automatic:
    return "automatic";
  default:
    return "unknown";
  }
}

// Forward declaration.
class NCCLBackend;

/**
 * Communicator for NCCL-based allreduces.
 * This requires NCCL version 2.0 or higher.
 */
class NCCLCommunicator : public CUDACommunicator {
  friend class NCCLBackend;
 public:
  /**
   *  Initialize a NCCL communicator on the world comm and default stream.
   */
  NCCLCommunicator() : NCCLCommunicator(MPI_COMM_WORLD, 0) {}
  /**
   * Initialize a NCCL communicator on the world comm and given stream.
   */
  NCCLCommunicator(cudaStream_t stream_) : NCCLCommunicator(MPI_COMM_WORLD, stream_) {}
  /**
   * Initialize a NCCL communicator.
   * @param comm_ An MPI_Comm representing the nodes to be in the communicator.
   * @param stream_ The stream to associate with the communicator.
   */
  NCCLCommunicator(MPI_Comm comm_, cudaStream_t stream_);
  ~NCCLCommunicator() override;
  Communicator* copy() const override {
    return new NCCLCommunicator(get_comm(), get_stream());
  }

 private:
  /** Initialize the internal NCCL communicator. */
  void nccl_setup();
  /** Clean up NCCL. */
  void nccl_destroy();

  /** NCCL communicator. */
  ncclComm_t m_nccl_comm;
};

namespace internal {
namespace nccl {

/** Initialize NCCL backend. */
void init(int& argc, char**& argv);
/** Finalize NCCL backend. */
void finalize();

/** Convert a ReductionOperator to the corresponding ncclRedOp_t. */
inline ncclRedOp_t ReductionOperator2ncclRedOp(ReductionOperator op) {
  switch(op) {
  case ReductionOperator::sum:
    return ncclSum;
  case ReductionOperator::prod:
    return ncclProd;
  case ReductionOperator::min:
    return ncclMin;
  case ReductionOperator::max:
    return ncclMax;
  default:
    throw_al_exception("Reduction operator not supported");
  }
}
template <typename T>
inline ncclDataType_t TypeMap();
template <> inline ncclDataType_t TypeMap<char>() { return ncclChar; }
template <> inline ncclDataType_t TypeMap<unsigned char>() { return ncclUint8; }
template <> inline ncclDataType_t TypeMap<int>() { return ncclInt; }
template <> inline ncclDataType_t TypeMap<unsigned int>() { return ncclUint32; }
template <> inline ncclDataType_t TypeMap<long long int>() { return ncclInt64; }
template <> inline ncclDataType_t TypeMap<unsigned long long int>() { return ncclUint64; }
template <> inline ncclDataType_t TypeMap<__half>() { return ncclHalf; }
template <> inline ncclDataType_t TypeMap<float>() { return ncclFloat; }
template <> inline ncclDataType_t TypeMap<double>() { return ncclDouble; }

/** Represents a request for the NCCL backend. */
struct NCCLRequest {
  NCCLRequest(cudaEvent_t op_event_, cudaStream_t orig_stream_) :
    op_event(op_event_), orig_stream(orig_stream_) {}
  // Note: Not thread safe!
  ~NCCLRequest() { cuda::release_cuda_event(op_event); }
  /** Event pending on completion of the operation. */
  cudaEvent_t op_event;
  /** Original stream associated with the operation. */
  cudaStream_t orig_stream;
};

}  // namespace nccl
}  // namespace internal

/** Backend implementing NCCL communication. */
class NCCLBackend {
  friend void internal::nccl::init(int&, char**&);
  friend void internal::nccl::finalize();
 public:
  using allreduce_algo_type = NCCLCollectiveAlgorithm;
  using bcast_algo_type = NCCLCollectiveAlgorithm;
  using reduce_algo_type = NCCLCollectiveAlgorithm;
  using allgather_algo_type = NCCLCollectiveAlgorithm;
  using reduce_scatter_algo_type = NCCLCollectiveAlgorithm;
  using comm_type = NCCLCommunicator;
  using req_type = std::shared_ptr<internal::nccl::NCCLRequest>;
  static constexpr std::nullptr_t null_req = nullptr;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type) {
    do_allreduce(sendbuf, recvbuf, count, op, comm, comm.get_stream());
  }

  template <typename T>
  static void Allreduce(T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type algo) {
    Allreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingAllreduce(const T* sendbuf, T* recvbuf, size_t count,
                                   ReductionOperator op, comm_type& comm,
                                   req_type& req, allreduce_algo_type) {
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_allreduce(sendbuf, recvbuf, count, op, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingAllreduce(T* recvbuf, size_t count,
                                   ReductionOperator op, comm_type& comm,
                                   req_type& req, allreduce_algo_type algo) {
    NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                         req, algo);
  }

  template <typename T>
  static void Bcast(T* buf, size_t count, int root, comm_type& comm,
                    bcast_algo_type) {
    do_broadcast(buf, count, root, comm, comm.get_stream());
  }

  template <typename T>
  static void NonblockingBcast(T* buf, size_t count, int root,
                               comm_type& comm, req_type& req, bcast_algo_type) {
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_broadcast(buf, count, root, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void Reduce(const T* sendbuf, T* recvbuf, size_t count,
                     ReductionOperator op, int root, comm_type& comm,
                     reduce_algo_type) {
    do_reduce(sendbuf, recvbuf, count, op, root, comm, comm.get_stream());
  }

  template <typename T>
  static void Reduce(T* recvbuf, size_t count,
                     ReductionOperator op, int root, comm_type& comm,
                     reduce_algo_type algo) {
    Reduce(internal::IN_PLACE<T>(), recvbuf, count, op, root, comm, algo);
  }

  template <typename T>
  static void NonblockingReduce(const T* sendbuf, T* recvbuf, size_t count,
                                ReductionOperator op, int root, comm_type& comm,
                                req_type& req, reduce_algo_type) {
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_reduce(sendbuf, recvbuf, count, op, root, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingReduce(T* recvbuf, size_t count, ReductionOperator op,
                                int root, comm_type& comm, req_type& req,
                                reduce_algo_type algo) {
    NonblockingReduce(internal::IN_PLACE<T>(), recvbuf, count, op, root, comm,
                       req, algo);
  }

  template <typename T>
  static void Allgather(const T* sendbuf, T* recvbuf, size_t send_count,
                        comm_type& comm, allgather_algo_type) {
    do_allgather(sendbuf, recvbuf, send_count, comm, comm.get_stream());
  }

  template <typename T>
  static void Allgather(T* recvbuf, size_t send_count,
                        comm_type& comm, allgather_algo_type algo) {
    Allgather(internal::IN_PLACE<T>(), recvbuf, send_count, comm, algo);
  }

  template <typename T>
  static void NonblockingAllgather(const T* sendbuf, T* recvbuf,
                                   size_t send_count, comm_type& comm,
                                   req_type& req, allgather_algo_type) {
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_allgather(sendbuf, recvbuf, send_count, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingAllgather(T* recvbuf, size_t send_count,
                                   comm_type& comm, req_type& req,
                                   allgather_algo_type algo) {
    NonblockingAllgather(internal::IN_PLACE<T>(), recvbuf, send_count, comm,
                         req, algo);
  }

  template <typename T>
  static void Reduce_scatter(const T* sendbuf, T* recvbuf, size_t count,
                             ReductionOperator op, comm_type& comm,
                             reduce_scatter_algo_type) {
    do_reduce_scatter(sendbuf, recvbuf, count, op, comm,
                      comm.get_stream());
  }

  template <typename T>
  static void Reduce_scatter(T* recvbuf, size_t count,
                             ReductionOperator op, comm_type& comm,
                             reduce_scatter_algo_type algo) {
    Reduce_scatter(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                   algo);
  }

  template <typename T>
  static void NonblockingReduce_scatter(const T* sendbuf, T* recvbuf,
                                        size_t count,
                                        ReductionOperator op, comm_type& comm,
                                        req_type& req, reduce_scatter_algo_type) {
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_reduce_scatter(sendbuf, recvbuf, count, op, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingReduce_scatter(T* recvbuf, size_t count,
                                        ReductionOperator op, comm_type& comm,
                                        req_type& req, reduce_scatter_algo_type algo) {
    NonblockingReduce_scatter(internal::IN_PLACE<T>(), recvbuf, count, op,
                              comm, req, algo);
  }

  static std::string Name() { return "NCCLBackend"; }

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
    req = std::make_shared<internal::nccl::NCCLRequest>(event, comm.get_stream());
  }
  // These are thin wrappers around the actual NCCL calls.
  /** Do a NCCL allreduce. */
  template <typename T>
  static void do_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                           ReductionOperator op, comm_type& comm,
                           cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    AL_CHECK_NCCL(ncclAllReduce((const void*) sendbuf, (void*) recvbuf, count,
                                internal::nccl::TypeMap<T>(),
                                internal::nccl::ReductionOperator2ncclRedOp(op),
                                comm.m_nccl_comm, stream));
  }
  /** Do a NCCL broadcast. */
  template <typename T>
  static void do_broadcast(T* buf, size_t count, int root, comm_type& comm,
                           cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    AL_CHECK_NCCL(ncclBroadcast((const void*) buf, (void*) buf, count,
                                internal::nccl::TypeMap<T>(), root,
                                comm.m_nccl_comm, stream));
  }
  /** Do a NCCL reduce. */
  template <typename T>
  static void do_reduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, int root, comm_type& comm,
                        cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    AL_CHECK_NCCL(ncclReduce((const void*) sendbuf, (void*) recvbuf, count,
                             internal::nccl::TypeMap<T>(),
                             internal::nccl::ReductionOperator2ncclRedOp(op),
                             root, comm.m_nccl_comm, stream));
  }
  /** Do a NCCL allgather. */
  template <typename T>
  static void do_allgather(const T* sendbuf, T* recvbuf, size_t send_count,
                           comm_type& comm, cudaStream_t stream) {
    if (send_count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf + comm.rank()*send_count;
    }
    AL_CHECK_NCCL(ncclAllGather((const void*) sendbuf, (void*) recvbuf,
                                send_count, internal::nccl::TypeMap<T>(),
                                comm.m_nccl_comm, stream));
  }
  /** Do a NCCL reduce-scatter. */
  template <typename T>
  static void do_reduce_scatter(const T* sendbuf, T* recvbuf,
                                size_t recv_count, ReductionOperator op,
                                comm_type& comm, cudaStream_t stream) {
    if (recv_count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    AL_CHECK_NCCL(ncclReduceScatter((const void*) sendbuf, (void*) recvbuf,
                                    recv_count, internal::nccl::TypeMap<T>(),
                                    internal::nccl::ReductionOperator2ncclRedOp(op),
                                    comm.m_nccl_comm, stream));
  }
};

template <>
inline bool Test<NCCLBackend>(typename NCCLBackend::req_type& req) {
  if (req == NCCLBackend::null_req) {
    return true;
  }
  // This is purely a host operation.
  bool r = cudaEventQuery(req->op_event) == cudaSuccess;
  if (r) {
    req = NCCLBackend::null_req;
  }
  return r;
}

template <>
inline void Wait<NCCLBackend>(typename NCCLBackend::req_type& req) {
  if (req == NCCLBackend::null_req) {
    return;
  }
  // Synchronize the original stream with the request.
  // This will not block the host.
  AL_CHECK_CUDA(cudaStreamWaitEvent(req->orig_stream, req->op_event, 0));
}

}  // namespace Al
