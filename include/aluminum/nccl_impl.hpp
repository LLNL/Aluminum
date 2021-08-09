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

#include <Al_config.hpp>

#if defined(AL_HAS_ROCM)
#include <rccl.h>
#elif defined(AL_HAS_CUDA)
#include <nccl.h>
#endif // defined(AL_HAS_ROCM)

#include "Al.hpp"
#include "aluminum/internal.hpp"
#include "aluminum/mempool.hpp"
#include "aluminum/cuda/cuda.hpp"
#include "aluminum/cuda/events.hpp"
#include "aluminum/cuda/streams.hpp"
#include "aluminum/mpi_comm_and_stream_wrapper.hpp"

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

/** Communicator for NCCL operations. */
class NCCLCommunicator : public internal::MPICommAndStreamWrapper<cudaStream_t> {
  friend class NCCLBackend;
 public:
  /** Default constructor, uses MPI_COMM_WORLD and the default stream. */
   NCCLCommunicator() : NCCLCommunicator(MPI_COMM_WORLD, 0) {}
  /** Use a particular MPI communicator and stream. */
  NCCLCommunicator(MPI_Comm comm_, cudaStream_t stream_ = 0);
  /** disable mpi. */
  NCCLCommunicator(int rank_, int size_, ncclUniqueId nccl_id_, cudaStream_t stream_ = 0);
  /** Cannot copy this. */
  NCCLCommunicator(const NCCLCommunicator& other) = delete;
  /** Default move constructor. */
  NCCLCommunicator(NCCLCommunicator&& other) = default;
  /** Cannot copy this. */
  NCCLCommunicator& operator=(const NCCLCommunicator& other) = delete;
  /** Default move assignment operator. */
  NCCLCommunicator& operator=(NCCLCommunicator&& other) = default;
  ~NCCLCommunicator();

  /** Create a new NCCLCommunicator with the same processes and a new stream. */
  NCCLCommunicator copy(cudaStream_t stream = 0) {
    return NCCLCommunicator(get_comm(), stream);
  }

  /** gracefully abort uncompleted nccl operations */
  void abort() {
      AL_CHECK_NCCL(ncclCommAbort(m_nccl_comm));
  }

 private:
  /** Raw NCCL communicator. */
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
  case ReductionOperator::avg:
    return ncclAvg;
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
  NCCLRequest(cudaEvent_t op_event_, cudaStream_t orig_stream_,
              cudaStream_t internal_stream_) :
    op_event(op_event_), orig_stream(orig_stream_),
    internal_stream(internal_stream_) {}
  ~NCCLRequest() { cuda::event_pool.release(op_event); }
  /** Event pending on completion of the operation. */
  cudaEvent_t op_event;
  /** Original stream associated with the operation. */
  cudaStream_t orig_stream;
  /** Internal stream the operation is running on. */
  cudaStream_t internal_stream;
};

/**
 * Safely execute a loop of grouped NCCL operations within operation limits.
 *
 * NCCL groups can contain only a limited number of operations before they
 * start generating errors. The maximum is hardcoded in NCCL and currently 2048
 * operations.
 *
 * This will safely split execution into multiple groups that stay within the
 * limit. It will execute a for loop beginning with start and ending with limit
 * and run func each iteration, with the current iteration value as an argument.
 * pre_func and post_func will be executed before or after the loop and can also
 * perform NCCL operations.
 *
 * @param start Initial value of the for loop.
 * @param limit Maximum value of the for loop.
 * @param func One iteration of the for loop.
 * @param pre_func Function to execute before for loop begins.
 * @param post_func Function to execute after for loop completes.
 * @param func_cond Only execute iterations of function if this returns true.
 * @tparam num_per_func Maximum number of NCCL calls per iteration.
 * @tparam num_per_pre Maximum number of NCCL calls in pre_func.
 * @tparam num_per_post Maximum number of NCCL calls in post_func.
 * @tparam max_per_group Maximum number of NCCL calls to allow in a group.
 */
template <size_t num_per_func,
          size_t num_per_pre = 0, size_t num_per_post = 0,
          size_t max_per_group = 1024>
void safe_nccl_group(size_t start, size_t limit,
                     std::function<void(int)> func,
                     std::function<void()> pre_func = [](){},
                     std::function<void()> post_func = [](){},
                     std::function<bool()> func_cond = [](){return true;}) {
  static_assert(num_per_func <= max_per_group,
                "Cannot have more NCCL calls per iteration than permitted");
  static_assert(num_per_pre <= max_per_group,
                "Cannot have more NCCL calls in pre than permitted");
  static_assert(num_per_post <= max_per_group,
                "Cannot have more NCCL calls in post than permitted");
  size_t cur_nccl_calls = 0;
  // Need this flag to ensure we close this group if no calls are actually made.
  bool closed = false;
  AL_CHECK_NCCL(ncclGroupStart());
  pre_func();
  if (num_per_pre == max_per_group) {
    // If there are exactly max_per_group calls, we can keep the count at 0,
    // since we need a new NCCL group anyway.
    AL_CHECK_NCCL(ncclGroupEnd());
    AL_CHECK_NCCL(ncclGroupStart());
  } else {
    cur_nccl_calls += num_per_pre;
  }

  if (func_cond()) {
    for (size_t i = start; i < limit; ++i) {
      if (cur_nccl_calls + num_per_func > max_per_group) {
        // Will exceed number of ops per group this iteration, so we close
        // the existing group and start a new one.
        AL_CHECK_NCCL(ncclGroupEnd());
        AL_CHECK_NCCL(ncclGroupStart());
        closed = true;
        cur_nccl_calls = 0;
      }
      func(i);
      cur_nccl_calls += num_per_func;
    }
  }

  if (cur_nccl_calls + num_per_post > max_per_group) {
    // Need to close the NCCL group and start a new one.
    AL_CHECK_NCCL(ncclGroupEnd());
    AL_CHECK_NCCL(ncclGroupStart());
    closed = true;
    cur_nccl_calls = 0;
  }
  post_func();
  cur_nccl_calls += num_per_post;
  if (cur_nccl_calls > 0 || !closed) {
    AL_CHECK_NCCL(ncclGroupEnd());
  }
}

}  // namespace nccl
}  // namespace internal

/** Backend implementing NCCL communication. */
class NCCLBackend {
  friend void internal::nccl::init(int&, char**&);
  friend void internal::nccl::finalize();
 public:
  using allreduce_algo_type = NCCLCollectiveAlgorithm;
  using barrier_algo_type = NCCLCollectiveAlgorithm;
  using bcast_algo_type = NCCLCollectiveAlgorithm;
  using gather_algo_type = NCCLCollectiveAlgorithm;
  using gatherv_algo_type = NCCLCollectiveAlgorithm;
  using reduce_algo_type = NCCLCollectiveAlgorithm;
  using allgather_algo_type = NCCLCollectiveAlgorithm;
  using allgatherv_algo_type = NCCLCollectiveAlgorithm;
  using alltoall_algo_type = NCCLCollectiveAlgorithm;
  using alltoallv_algo_type = NCCLCollectiveAlgorithm;
  using reduce_scatter_algo_type = NCCLCollectiveAlgorithm;
  using reduce_scatterv_algo_type = NCCLCollectiveAlgorithm;
  using scatter_algo_type = NCCLCollectiveAlgorithm;
  using scatterv_algo_type = NCCLCollectiveAlgorithm;
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
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
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
  static void Send(const T* sendbuf, size_t count, int dest, comm_type& comm) {
    do_send(sendbuf, count, dest, comm, comm.get_stream());
  }

  template <typename T>
  static void NonblockingSend(const T* sendbuf, size_t count, int dest,
                              comm_type& comm, req_type& req) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_send(sendbuf, count, dest, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void Recv(T* recvbuf, size_t count, int src, comm_type& comm) {
    do_recv(recvbuf, count, src, comm, comm.get_stream());
  }

  template <typename T>
  static void NonblockingRecv(T* recvbuf, size_t count, int src,
                              comm_type& comm, req_type& req) {
      cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
      sync_internal_stream_with_comm(internal_stream, comm);
      do_recv(recvbuf, count, src, comm, internal_stream);
      setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void SendRecv(const T* sendbuf, size_t send_count, int dest,
                       T* recvbuf, size_t recv_count, int src, comm_type& comm) {
    do_sendrecv(sendbuf, send_count, dest, recvbuf, recv_count, src,
                comm, comm.get_stream());
  }

  template <typename T>
  static void NonblockingSendRecv(const T* sendbuf, size_t send_count, int dest,
                                  T* recvbuf, size_t recv_count, int src,
                                  comm_type& comm, req_type& req) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_sendrecv(sendbuf, send_count, dest, recvbuf, recv_count, src,
                comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  static void Barrier(comm_type& comm, barrier_algo_type) {
    do_barrier(comm, comm.get_stream());
  }

  static void NonblockingBarrier(comm_type& comm, req_type& req,
                                 barrier_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_barrier(comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void Bcast(T* buf, size_t count, int root, comm_type& comm,
                    bcast_algo_type) {
    do_broadcast(buf, count, root, comm, comm.get_stream());
  }

  template <typename T>
  static void NonblockingBcast(T* buf, size_t count, int root,
                               comm_type& comm, req_type& req, bcast_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_broadcast(buf, count, root, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void Gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                     comm_type& comm, gather_algo_type) {
    do_gather(sendbuf, recvbuf, count, root, comm, comm.get_stream());
  }

  template <typename T>
  static void Gather(T* buffer, size_t count, int root, comm_type& comm,
                     gather_algo_type algo) {
    Gather(internal::IN_PLACE<T>(), buffer, count, root, comm, algo);
  }

  template <typename T>
  static void NonblockingGather(const T* sendbuf, T* recvbuf, size_t count,
                                int root, comm_type& comm, req_type& req,
                                gather_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_gather(sendbuf, recvbuf, count, root, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingGather(T* buffer, size_t count, int root,
                                comm_type& comm, req_type& req,
                                gather_algo_type algo) {
    NonblockingGather(internal::IN_PLACE<T>(), buffer, count, root, comm, req, algo);
  }

  template <typename T>
  static void Gatherv(const T* sendbuf, T* recvbuf,
                      std::vector<size_t> counts,
                      std::vector<size_t> displs,
                      int root, comm_type& comm, gatherv_algo_type) {
    do_gatherv(sendbuf, recvbuf, counts, displs, root, comm, comm.get_stream());
  }

  template <typename T>
  static void Gatherv(T* buffer,
                      std::vector<size_t> counts,
                      std::vector<size_t> displs,
                      int root, comm_type& comm, gatherv_algo_type algo) {
    Gatherv(internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, algo);
  }

  template <typename T>
  static void NonblockingGatherv(const T* sendbuf, T* recvbuf,
                                 std::vector<size_t> counts,
                                 std::vector<size_t> displs,
                                 int root, comm_type& comm, req_type& req,
                                 gatherv_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_gatherv(sendbuf, recvbuf, counts, displs, root, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingGatherv(T* buffer,
                                 std::vector<size_t> counts,
                                 std::vector<size_t> displs,
                                 int root, comm_type& comm, req_type& req,
                                 gatherv_algo_type algo) {
    NonblockingGatherv(internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, req, algo);
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
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
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
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
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
  static void Allgatherv(const T* sendbuf, T* recvbuf,
                         std::vector<size_t> counts,
                         std::vector<size_t> displs,
                         comm_type& comm, allgatherv_algo_type) {
    do_allgatherv(sendbuf, recvbuf, counts, displs, comm, comm.get_stream());
  }

  template <typename T>
  static void Allgatherv(T* buffer,
                         std::vector<size_t> counts,
                         std::vector<size_t> displs,
                         comm_type& comm, allgatherv_algo_type algo) {
    Allgatherv(internal::IN_PLACE<T>(), buffer, counts, displs, comm, algo);
  }

  template <typename T>
  static void NonblockingAllgatherv(const T* sendbuf, T* recvbuf,
                                    std::vector<size_t> counts,
                                    std::vector<size_t> displs,
                                    comm_type& comm, req_type& req,
                                    allgatherv_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_allgatherv(sendbuf, recvbuf, counts, displs, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingAllgatherv(T* buffer,
                                    std::vector<size_t> counts,
                                    std::vector<size_t> displs,
                                    comm_type& comm, req_type& req,
                                    allgatherv_algo_type algo) {
    NonblockingAllgatherv(internal::IN_PLACE<T>(), buffer, counts, displs,
                          comm, req, algo);
  }

  template <typename T>
  static void Alltoall(const T* sendbuf, T* recvbuf, size_t count,
                       comm_type& comm, alltoall_algo_type) {
    do_alltoall(sendbuf, recvbuf, count, comm, comm.get_stream());
  }

  template <typename T>
  static void Alltoall(T* buffer, size_t count, comm_type& comm,
                       alltoall_algo_type algo) {
    Alltoall(internal::IN_PLACE<T>(), buffer, count, comm, algo);
  }

  template <typename T>
  static void NonblockingAlltoall(const T* sendbuf, T* recvbuf, size_t count,
                                  comm_type& comm, req_type& req,
                                  alltoall_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_alltoall(sendbuf, recvbuf, count, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingAlltoall(T* buffer, size_t count, comm_type& comm,
                                  req_type& req, alltoall_algo_type algo) {
    NonblockingAlltoall(internal::IN_PLACE<T>(), buffer, count, comm, req, algo);
  }

  template <typename T>
  static void Alltoallv(const T* sendbuf,
                        std::vector<size_t> send_counts,
                        std::vector<size_t> send_displs,
                        T* recvbuf,
                        std::vector<size_t> recv_counts,
                        std::vector<size_t> recv_displs,
                        comm_type& comm,
                        alltoallv_algo_type) {
    do_alltoallv(sendbuf, send_counts, send_displs,
                 recvbuf, recv_counts, recv_displs,
                 comm, comm.get_stream());
  }

  template <typename T>
  static void Alltoallv(T* buffer,
                        std::vector<size_t> counts,
                        std::vector<size_t> displs,
                        comm_type& comm,
                        alltoallv_algo_type algo) {
    Alltoallv(internal::IN_PLACE<T>(), counts, displs, buffer, counts, displs,
              comm, algo);
  }

  template <typename T>
  static void NonblockingAlltoallv(const T* sendbuf,
                                   std::vector<size_t> send_counts,
                                   std::vector<size_t> send_displs,
                                   T* recvbuf,
                                   std::vector<size_t> recv_counts,
                                   std::vector<size_t> recv_displs,
                                   comm_type& comm, req_type& req,
                                   alltoallv_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_alltoallv(sendbuf, send_counts, send_displs,
                 recvbuf, recv_counts, recv_displs,
                 comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingAlltoallv(T* buffer,
                                   std::vector<size_t> counts,
                                   std::vector<size_t> displs,
                                   comm_type& comm, req_type& req,
                                   alltoallv_algo_type algo) {
    NonblockingAlltoallv(internal::IN_PLACE<T>(), counts, displs,
                         buffer, counts, displs,
                         comm, req, algo);
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
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
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

  template <typename T>
  static void Reduce_scatterv(const T* sendbuf, T* recvbuf,
                              std::vector<size_t> counts,
                              ReductionOperator op, comm_type& comm,
                              reduce_scatter_algo_type) {
    do_reduce_scatterv(sendbuf, recvbuf, counts, op, comm,
                       comm.get_stream());
  }

  template <typename T>
  static void Reduce_scatterv(T* recvbuf, std::vector<size_t> counts,
                              ReductionOperator op, comm_type& comm,
                              reduce_scatter_algo_type algo) {
    Reduce_scatterv(internal::IN_PLACE<T>(), recvbuf, counts, op, comm,
                    algo);
  }

  template <typename T>
  static void NonblockingReduce_scatterv(const T* sendbuf, T* recvbuf,
                                         std::vector<size_t> counts,
                                         ReductionOperator op,
                                         comm_type& comm,
                                         req_type& req,
                                         reduce_scatterv_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_reduce_scatterv(sendbuf, recvbuf, counts, op, comm,
                       internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingReduce_scatterv(T* recvbuf,
                                         std::vector<size_t> counts,
                                         ReductionOperator op,
                                         comm_type& comm,
                                         req_type& req,
                                         reduce_scatter_algo_type algo) {
    NonblockingReduce_scatterv(
      internal::IN_PLACE<T>(), recvbuf, counts, op, comm, req, algo);
  }

  template <typename T>
  static void Scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                      comm_type& comm, scatter_algo_type) {
    do_scatter(sendbuf, recvbuf, count, root, comm, comm.get_stream());
  }

  template <typename T>
  static void Scatter(T* buffer, size_t count, int root,
                      comm_type& comm, scatter_algo_type algo) {
    Scatter(internal::IN_PLACE<T>(), buffer, count, root, comm, algo);
  }

  template <typename T>
  static void NonblockingScatter(const T* sendbuf, T* recvbuf, size_t count,
                                 int root, comm_type& comm, req_type& req,
                                 scatter_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_scatter(sendbuf, recvbuf, count, root, comm, internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingScatter(T* buffer, size_t count, int root,
                                 comm_type& comm, req_type& req,
                                 scatter_algo_type algo) {
    NonblockingScatter(internal::IN_PLACE<T>(), buffer, count, root, comm, req, algo);
  }

  template <typename T>
  static void Scatterv(const T* sendbuf, T* recvbuf,
                       std::vector<size_t> counts,
                       std::vector<size_t> displs,
                       int root, comm_type& comm,
                       scatterv_algo_type) {
    do_scatterv(sendbuf, recvbuf, counts, displs, root, comm,
                comm.get_stream());
  }

  template <typename T>
  static void Scatterv(T* buffer,
                       std::vector<size_t> counts,
                       std::vector<size_t> displs,
                       int root, comm_type& comm,
                       scatterv_algo_type algo) {
    Scatterv(internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, algo);
  }

  template <typename T>
  static void NonblockingScatterv(const T* sendbuf, T* recvbuf,
                                  std::vector<size_t> counts,
                                  std::vector<size_t> displs,
                                  int root, comm_type& comm, req_type& req,
                                  scatterv_algo_type) {
    cudaStream_t internal_stream = internal::cuda::stream_pool.get_high_priority_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    do_scatterv(sendbuf, recvbuf, counts, displs, root, comm,
                internal_stream);
    setup_completion_event(internal_stream, comm, req);
  }

  template <typename T>
  static void NonblockingScatterv(T* buffer,
                                  std::vector<size_t> counts,
                                  std::vector<size_t> displs,
                                  int root, comm_type& comm, req_type& req,
                                  scatterv_algo_type algo) {
    NonblockingScatterv(internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, req, algo);
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
    cudaEvent_t event = internal::cuda::event_pool.get();
    AL_CHECK_CUDA(cudaEventRecord(event, internal_stream));
    req = std::make_shared<internal::nccl::NCCLRequest>(
      event, comm.get_stream(), internal_stream);
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

  /** Do a NCCL send. */
  template <typename T>
  static void do_send(const T* sendbuf, size_t count, int dest, comm_type& comm,
                      cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    AL_CHECK_NCCL(ncclSend((const void*) sendbuf, count,
                           internal::nccl::TypeMap<T>(), dest,
                           comm.m_nccl_comm, stream));
  }

  /** Do a NCCL recv. */
  template <typename T>
  static void do_recv(T* recvbuf, size_t count, int src, comm_type& comm,
                      cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    AL_CHECK_NCCL(ncclRecv((void*) recvbuf, count,
                           internal::nccl::TypeMap<T>(), src,
                           comm.m_nccl_comm, stream));
  }

  /** Do a NCCL sendrecv. */
  template <typename T>
  static void do_sendrecv(const T* sendbuf, size_t send_count, int dest,
                          T* recvbuf, size_t recv_count, int src,
                          comm_type& comm, cudaStream_t stream) {
    if (send_count == 0 && recv_count == 0) {
      return;
    }
    AL_CHECK_NCCL(ncclGroupStart());
    // Work around some sort of potential bug.
    if (send_count != 0) {
      AL_CHECK_NCCL(ncclSend((const void*) sendbuf, send_count,
                             internal::nccl::TypeMap<T>(), dest,
                             comm.m_nccl_comm, stream));
    }
    if (recv_count != 0) {
      AL_CHECK_NCCL(ncclRecv((void*) recvbuf, recv_count,
                             internal::nccl::TypeMap<T>(), src,
                             comm.m_nccl_comm, stream));
    }
    AL_CHECK_NCCL(ncclGroupEnd());
  }

  /** Do a NCCL barrier. */
  static void do_barrier(comm_type& comm, cudaStream_t stream) {
    // Implement the barrier as an allreduce on a single value.
    using barrier_t = unsigned char;
    barrier_t* barrier_buf =
      internal::mempool.allocate<internal::MemoryType::CUDA, barrier_t>(1, stream);
    AL_CHECK_NCCL(ncclAllReduce(
                    (const void*) barrier_buf, (void*) barrier_buf, 1,
                    internal::nccl::TypeMap<barrier_t>(), ncclSum,
                    comm.m_nccl_comm, stream));
    internal::mempool.release<internal::MemoryType::CUDA>(barrier_buf);
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

  /** Do a NCCL gather. */
  template <typename T>
  static void do_gather(const T* sendbuf, T* recvbuf, size_t count, int root,
                        comm_type& comm, cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>()) {
      if (comm.rank() == root) {
        sendbuf = recvbuf + comm.rank() * count;
      } else {
        sendbuf = recvbuf;
      }
    }
    internal::nccl::safe_nccl_group<1, 0, 2>(
      0, comm.size(),
      [&](int rank) {
        // The send/recv to/from self must be bundled in the same group.
        if (rank == root) { return; }
        AL_CHECK_NCCL(ncclRecv((void*) &recvbuf[rank*count], count,
                               internal::nccl::TypeMap<T>(), rank,
                               comm.m_nccl_comm, stream));
      },
      [](){},
      [&]() {
        AL_CHECK_NCCL(ncclSend((const void*) sendbuf, count,
                               internal::nccl::TypeMap<T>(), root,
                               comm.m_nccl_comm, stream));
        if (comm.rank() == root) {
          AL_CHECK_NCCL(ncclRecv((void*) &recvbuf[root*count], count,
                                 internal::nccl::TypeMap<T>(), root,
                                 comm.m_nccl_comm, stream));
        }
      },
      [&]() { return comm.rank() == root; });
  }

  /** Do a NCCL vector gather. */
  template <typename T>
  static void do_gatherv(const T* sendbuf, T* recvbuf,
                         std::vector<size_t> counts, std::vector<size_t> displs,
                         int root, comm_type& comm, cudaStream_t stream) {
    if (sendbuf == internal::IN_PLACE<T>()) {
      if (comm.rank() == root) {
        sendbuf = recvbuf + displs[comm.rank()];
      } else {
        sendbuf = recvbuf;
      }
    }
    internal::nccl::safe_nccl_group<1, 0, 2>(
      0, comm.size(),
      [&](int rank) {
        // The send/recv to/from self must be handled in the same group.
        if (rank == root) { return; }
        if (counts[rank] == 0) { return; }
        AL_CHECK_NCCL(ncclRecv((void*) (recvbuf + displs[rank]), counts[rank],
                               internal::nccl::TypeMap<T>(), rank,
                               comm.m_nccl_comm, stream));
      },
      [](){},
      [&]() {
        if (counts[comm.rank()] == 0) { return; }
        AL_CHECK_NCCL(ncclSend((const void*) sendbuf, counts[comm.rank()],
                               internal::nccl::TypeMap<T>(), root,
                               comm.m_nccl_comm, stream));
        if (comm.rank() == root) {
          AL_CHECK_NCCL(ncclRecv((void*) (recvbuf + displs[root]), counts[root],
                                 internal::nccl::TypeMap<T>(), root,
                                 comm.m_nccl_comm, stream));
        }
      },
      [&]() { return comm.rank() == root; });
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

  /** Do a NCCL vector allgather. */
  template <typename T>
  static void do_allgatherv(const T* sendbuf, T* recvbuf,
                            std::vector<size_t> counts, std::vector<size_t> displs,
                            comm_type& comm, cudaStream_t stream) {
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf + displs[comm.rank()];
    }
    internal::nccl::safe_nccl_group<2>(
      0, comm.size(),
      [&](int rank) {
        if (counts[rank] > 0) {
          AL_CHECK_NCCL(ncclSend((const void*) sendbuf, counts[comm.rank()],
                                 internal::nccl::TypeMap<T>(), rank,
                                 comm.m_nccl_comm, stream));
          AL_CHECK_NCCL(ncclRecv((void*) (recvbuf + displs[rank]), counts[rank],
                                 internal::nccl::TypeMap<T>(), rank,
                                 comm.m_nccl_comm, stream));
        }
      });
  }

  /** Do a NCCL alltoall. */
  template <typename T>
  static void do_alltoall(const T* sendbuf, T* recvbuf, size_t count,
                          comm_type& comm, cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    T* tmp_sendbuf = const_cast<T*>(sendbuf);
    if (sendbuf == recvbuf) {
      tmp_sendbuf = internal::mempool.allocate<internal::MemoryType::CUDA, T>(
        count*comm.size(), stream);
      AL_CHECK_CUDA(cudaMemcpyAsync(tmp_sendbuf, sendbuf,
                                    count*sizeof(T)*comm.size(),
                                    cudaMemcpyDeviceToDevice, stream));
    }
    internal::nccl::safe_nccl_group<2>(
      0, comm.size(),
      [&](int rank) {
        AL_CHECK_NCCL(ncclSend((const void*) &tmp_sendbuf[rank*count], count,
                             internal::nccl::TypeMap<T>(), rank,
                             comm.m_nccl_comm, stream));
        AL_CHECK_NCCL(ncclRecv((void*) &recvbuf[rank*count], count,
                               internal::nccl::TypeMap<T>(), rank,
                               comm.m_nccl_comm, stream));
      });
    if (tmp_sendbuf != sendbuf) {
      internal::mempool.release<internal::MemoryType::CUDA>(tmp_sendbuf);
    }
  }

  /** Do a NCCL vector alltoall. */
  template <typename T>
  static void do_alltoallv(const T* sendbuf,
                           std::vector<size_t> send_counts,
                           std::vector<size_t> send_displs,
                           T* recvbuf,
                           std::vector<size_t> recv_counts,
                           std::vector<size_t> recv_displs,
                           comm_type& comm, cudaStream_t stream) {
    T* tmp_sendbuf = const_cast<T*>(sendbuf);
    if (sendbuf == internal::IN_PLACE<T>()) {
      // We send and receive to this buffer, so we need a temporary one.
      // For simplicity, we will ensure it is contiguous.
      // TODO: Optimize this.
      size_t sendbuf_len = std::accumulate(send_counts.begin(),
                                           send_counts.end(), 0);
      std::vector<size_t> contig_displs = excl_prefix_sum(send_counts);
      tmp_sendbuf = internal::mempool.allocate<internal::MemoryType::CUDA, T>(
        sendbuf_len, stream);
      // TODO: Optimize for the case where everything is contiguous.
      for (size_t i = 0; i < send_counts.size(); ++i) {
        AL_CHECK_CUDA(cudaMemcpyAsync((void*) (tmp_sendbuf + contig_displs[i]),
                                      (const void*) (recvbuf + send_displs[i]),
                                      send_counts[i]*sizeof(T),
                                      cudaMemcpyDeviceToDevice, stream));
      }
      // The copied data is contiguous.
      send_displs = contig_displs;
    }
    internal::nccl::safe_nccl_group<2>(
      0, comm.size(),
      [&](int rank) {
        if (send_counts[rank] > 0) {
          AL_CHECK_NCCL(ncclSend((const void*) (tmp_sendbuf + send_displs[rank]),
                                 send_counts[rank], internal::nccl::TypeMap<T>(),
                                 rank, comm.m_nccl_comm, stream));
        }
        if (recv_counts[rank] > 0) {
          AL_CHECK_NCCL(ncclRecv((void*) (recvbuf + recv_displs[rank]),
                                 recv_counts[rank], internal::nccl::TypeMap<T>(),
                                 rank, comm.m_nccl_comm, stream));
        }
      });
    if (tmp_sendbuf != sendbuf) {
      internal::mempool.release<internal::MemoryType::CUDA>(tmp_sendbuf);
    }
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

  /** Do a NCCL vector reduce-scatter. */
  template <typename T>
  static void do_reduce_scatterv(const T* sendbuf, T* recvbuf,
                                 std::vector<size_t> counts,
                                 ReductionOperator op,
                                 comm_type& comm,
                                 cudaStream_t stream) {
    // This is implemented as a reduce followed by a scatterv.
    // Rank 0 is the root.
    size_t count = std::accumulate(counts.begin(), counts.end(), 0);
    std::vector<size_t> displs = excl_prefix_sum(counts);
    // Need a temporary reduce buffer so we don't trash the entire thing.
    T* tmp_redbuf = internal::mempool.allocate<internal::MemoryType::CUDA, T>(
      count, stream);
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    do_reduce(sendbuf, tmp_redbuf, count, op, 0, comm, stream);
    do_scatterv(tmp_redbuf, recvbuf, counts, displs, 0, comm, stream);
    if (tmp_redbuf != recvbuf) {
      internal::mempool.release<internal::MemoryType::CUDA>(tmp_redbuf);
    }
  }

  /** Do a NCCL scatter. */
  template <typename T>
  static void do_scatter(const T* sendbuf, T* recvbuf, size_t count, int root,
                         comm_type& comm, cudaStream_t stream) {
    if (count == 0) {
      return;
    }
    if (sendbuf == internal::IN_PLACE<T>() && comm.rank() == root) {
      sendbuf = recvbuf;
      recvbuf = recvbuf + comm.rank()*count;
    }
    internal::nccl::safe_nccl_group<1, 0, 2>(
      0, comm.size(),
      [&](int rank) {
        // The send/recv to/from self must be bundled in the same group.
        if (rank == root) { return; }
        AL_CHECK_NCCL(ncclSend((const void*) &sendbuf[rank*count], count,
                               internal::nccl::TypeMap<T>(), rank,
                               comm.m_nccl_comm, stream));
      },
      [](){},
      [&]() {
        AL_CHECK_NCCL(ncclRecv((void*) recvbuf, count,
                               internal::nccl::TypeMap<T>(), root,
                               comm.m_nccl_comm, stream));
        if (comm.rank() == root) {
          AL_CHECK_NCCL(ncclSend((const void*) &sendbuf[root*count], count,
                                 internal::nccl::TypeMap<T>(), root,
                                 comm.m_nccl_comm, stream));
        }
      },
      [&]() { return comm.rank() == root; });
  }

  /** Do a NCCL vector scatter. */
  template <typename T>
  static void do_scatterv(const T* sendbuf, T* recvbuf,
                          std::vector<size_t> counts,
                          std::vector<size_t> displs,
                          int root, comm_type& comm, cudaStream_t stream) {
    if (sendbuf == internal::IN_PLACE<T>() && comm.rank() == root) {
      sendbuf = recvbuf;
      recvbuf = recvbuf + displs[comm.rank()];
    }
    internal::nccl::safe_nccl_group<1, 0, 2>(
      0, comm.size(),
      [&](int rank) {
        // The send/recv to/from self must be bundled in the same group.
        if (rank == root) { return; }
        if (counts[rank] == 0) { return; }
        AL_CHECK_NCCL(ncclSend((const void*) (sendbuf + displs[rank]),
                               counts[rank], internal::nccl::TypeMap<T>(),
                               rank, comm.m_nccl_comm, stream));
      },
      [](){},
      [&]() {
        if (counts[comm.rank()] == 0) { return; }
        AL_CHECK_NCCL(ncclRecv((void*) recvbuf, counts[comm.rank()],
                               internal::nccl::TypeMap<T>(), root,
                               comm.m_nccl_comm, stream));
        if (comm.rank() == root) {
          AL_CHECK_NCCL(ncclSend((const void*) (sendbuf + displs[root]),
                                 counts[root], internal::nccl::TypeMap<T>(),
                                 root, comm.m_nccl_comm, stream));
        }
      },
      [&]() { return comm.rank() == root; });
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
