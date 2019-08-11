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

/**
 * Various optimized collective implementations.
 */

#pragma once

#include <iostream>
#include <mpi.h>

#include "Al_config.hpp"
#include "base.hpp"
#include "tuning_params.hpp"
#include "utils.hpp"
#include "profiling.hpp"
#include "trace.hpp"

namespace Al {

/** HOST_NAME_MAX is a linux only define */
#ifndef HOST_NAME_MAX
# if defined(_POSIX_HOST_NAME_MAX)
#  define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
# elif defined(MAXHOSTNAMELEN)
#  define HOST_NAME_MAX MAXHOSTNAMELEN
# endif
#endif /* HOST_NAME_MAX */

/**
 * Abstract base class for all communicator objects.
 * Implementation note: Try to keep these lightweight.
 */
class Communicator {
 public:
  /** Basic empty constructor. */
  Communicator() {}
  /** Create a communicator based on an MPI communicator. */
  Communicator(MPI_Comm) {}
  /** Default copy constructor. */
  Communicator(const Communicator& other) = default;
  /** Default move constructor. */
  Communicator(Communicator&& other) = default;
  /** Default copy assignment operator. */
  Communicator& operator=(const Communicator& other) noexcept = default;
  /** Default move assignment operator. */
  Communicator& operator=(Communicator&& other) = default;

  /** Empty destructor. */
  virtual ~Communicator() noexcept {}
  /** Returns a copy of the concrete class. */
  virtual Communicator* copy() const = 0;
  /** Return the rank of the calling process in the communicator. */
  virtual int rank() const = 0;
  /** Return the number of processes in the communicator. */
  virtual int size() const = 0;
  /** Return the rank of this process on the node it is on. */
  virtual int local_rank() const = 0;
  /** Return the number of processes in the communicator on the same node. */
  virtual int local_size() const = 0;
};

/**
 * Initialize Aluminum.
 * This must be called before any other calls to the library. It is safe to
 * call this multiple times.
 */
void Initialize(int& argc, char**& argv);
/**
 * Clean up Aluminum.
 * Do not make any further calls to the library after calling this.
 */
void Finalize();
/** Return true if Aluminum has been initialized. */
bool Initialized();

/**
 * Perform an allreduce.
 * @param sendbuf Input data.
 * @param recvbuf Output data; should already be allocated.
 * @param count Length of sendbuf and recvbuf.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce over.
 * @param algo Request a particular allreduce algorithm.
 */
template <typename Backend, typename T>
void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
               ReductionOperator op, typename Backend::comm_type& comm,
               typename Backend::allreduce_algo_type algo =
               Backend::allreduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("allreduce", comm, sendbuf, recvbuf,
                                         count);
  Backend::template Allreduce<T>(sendbuf, recvbuf, count, op, comm, algo);
}

/**
 * Perform an in-place allreduce.
 * @param recvbuf Input and output data; input will be overwritten.
 * @param count Length of recvbuf.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce over.
 * @param algo Request a particular allreduce algorithm.
 */
template <typename Backend, typename T>
void Allreduce(T* recvbuf, size_t count,
               ReductionOperator op, typename Backend::comm_type& comm,
               typename Backend::allreduce_algo_type algo =
               Backend::allreduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("allreduce", comm, recvbuf, count);
  Backend::template Allreduce<T>(recvbuf, count, op, comm, algo);
}

/**
 * Non-blocking version of Allreduce.
 * This returns immediately (i.e. does only local operations) and starts the
 * allreduce asynchronously. The request object is set to an opaque reference
 * for the allreduce, and can be checked using Test and Wait.
 * It is not safe to modify sendbuf or recvbuf until the request indicates that
 * the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingAllreduce(
  const T* sendbuf, T* recvbuf, size_t count,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::allreduce_algo_type algo =
  Backend::allreduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-allreduce", comm, sendbuf,
                                         recvbuf, count);
  Backend::template NonblockingAllreduce<T>(sendbuf, recvbuf, count, op,
                                            comm, req, algo);
}
/** In-place version of NonblockingAllreduce; same semantics apply. */
template <typename Backend, typename T>
void NonblockingAllreduce(
  T* recvbuf, size_t count,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::allreduce_algo_type algo =
  Backend::allreduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-allreduce", comm,
                                         recvbuf, count);
  Backend::template NonblockingAllreduce<T>(recvbuf, count, op,
                                            comm, req, algo);
}

/**
 * Perform a reduction.
 * @param sendbuf Input data.
 * @param recvbuf Output data; should already be allocated.
 * @param count Length of sendbuf and recvbuf.
 * @param op The reduction operation to perform.
 * @param root The root of the reduction, which gets the final result.
 * @param comm The communicator to reduce over.
 * @param algo Request a particular reduction algorithm.
 */
template <typename Backend, typename T>
void Reduce(const T* sendbuf, T* recvbuf, size_t count,
            ReductionOperator op, int root, typename Backend::comm_type& comm,
            typename Backend::reduce_algo_type algo =
            Backend::reduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("reduce", comm, sendbuf, recvbuf,
                                         count, root);
  Backend::template Reduce<T>(sendbuf, recvbuf, count, op, root, comm, algo);
}

/**
 * Perform an in-place reduction.
 * @param recvbuf Input and output data; input will be overwritten.
 * @param count Length of recvbuf.
 * @param op The reduction operation to perform.
 * @param root The root of the reduction, which gets the final result.
 * @param comm The communicator to reduce over.
 * @param algo Request a particular reduction algorithm.
 */
template <typename Backend, typename T>
void Reduce(T* recvbuf, size_t count,
            ReductionOperator op, int root, typename Backend::comm_type& comm,
            typename Backend::reduce_algo_type algo =
            Backend::reduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("reduce", comm, recvbuf, count, root);
  Backend::template Reduce<T>(recvbuf, count, op, root, comm, algo);
}

/**
 * Non-blocking version of Reduce.
 * This returns immediately (i.e. does only local operations) and starts the
 * reduction asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request indicates that
 * the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingReduce(
  const T* sendbuf, T* recvbuf, size_t count,
  ReductionOperator op,
  int root,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::reduce_algo_type algo =
  Backend::reduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-reduce", comm, sendbuf,
                                         recvbuf, count, root);
  Backend::template NonblockingReduce<T>(sendbuf, recvbuf, count, op, root, comm, req, algo);
}

/** In-place version of NonblockingReduce; same semantics apply. */
template <typename Backend, typename T>
void NonblockingReduce(
  T* recvbuf, size_t count,
  ReductionOperator op,
  int root,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::reduce_algo_type algo =
  Backend::reduce_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-reduce", comm, recvbuf,
                                         count, root);
  Backend::template NonblockingReduce<T>(recvbuf, count, op, root, comm, req, algo);
}

/**
 * @brief Perform a reduce-scatter.
 *
 * This is analogous to "MPI_Reduce_scatter_block" and matches NCCL's
 * interface.
 *
 * @param sendbuf Input data.
 * @param recvbuf Output data; should already be allocated.
 * @param count Length of recvbuf.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce/scatter over.
 * @param algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(
  const T* sendbuf, T* recvbuf, size_t count,
  ReductionOperator op, typename Backend::comm_type& comm,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("reduce_scatter", comm, sendbuf,
                                         recvbuf, count);
  Backend::template Reduce_scatter<T>(sendbuf, recvbuf, count, op, comm, algo);
}

/**
 * @brief Perform an in-place reduce-scatter.
 *
 * This is analogous to "MPI_Reduce_scatter_block" and matches NCCL's
 * interface.
 *
 * @param recvbuf Input and output data; input will be overwritten.
 * @param count Length of data to be received.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce/scatter over.
 * @param algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(
  T* recvbuf, size_t count,
  ReductionOperator op, typename Backend::comm_type& comm,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("reduce_scatter", comm, recvbuf, count);
  Backend::template Reduce_scatter<T>(recvbuf, count, op, comm, algo);
}

/**
 * Perform a reduce-scatter.
 * @param sendbuf Input data.
 * @param recvbuf Output data; should already be allocated.
 * @param counts Length of recvbuf.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce/scatter over.
 * @param algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(
  const T* sendbuf, T* recvbuf, size_t *counts,
  ReductionOperator op, typename Backend::comm_type& comm,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  Backend::template Reduce_scatter<T>(sendbuf, recvbuf, counts, op, comm, algo);
}

/**
 * Perform an in-place reduce-scatter.
 * @param recvbuf Input and output data; input will be overwritten.
 * @param counts Length of data to be received.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce/scatter over.
 * @param algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(
  T* recvbuf, size_t *counts,
  ReductionOperator op, typename Backend::comm_type& comm,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  Backend::template Reduce_scatter<T>(recvbuf, counts, op, comm, algo);
}

/**
 * @brief Non-blocking version of Reduce_scatter.
 *
 * This is analogous to "MPI_Ireduce_scatter_block" and matches NCCL's
 * interface.
 *
 * This returns immediately (i.e. does only local operations) and starts the
 * reduce-scatter asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request indicates that
 * the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
  const T* sendbuf, T* recvbuf, size_t count,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-reduce_scatter", comm,
                                         sendbuf, recvbuf, count);
  Backend::template NonblockingReduce_scatter<T>(
    sendbuf, recvbuf, count, op, comm, req, algo);
}

/** @brief In-place version of NonblockingReduce_scatter.
 *
 *  The same semantics apply.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
  T* recvbuf, size_t count,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-reduce_scatter", comm,
                                         recvbuf, count);
  Backend::template NonblockingReduce_scatter<T>(recvbuf, count, op, comm, req, algo);
}

/**
 * Non-blocking version of Reduce_scatter.
 * This returns immediately (i.e. does only local operations) and starts the
 * reduce-scatter asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request indicates that
 * the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
  const T* sendbuf, T* recvbuf, size_t *counts,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  Backend::template NonblockingReduce_scatter<T>(
    sendbuf, recvbuf, counts, op, comm, req, algo);
}

/** In-place version of NonblockingReduce_scatter; same semantics apply. */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
  T* recvbuf, size_t *counts,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::reduce_scatter_algo_type algo =
  Backend::reduce_scatter_algo_type::automatic) {
  Backend::template NonblockingReduce_scatter<T>(recvbuf, counts, op, comm, req, algo);
}

/**
 * Perform an allgather.
 * @param sendbuf Input data.
 * @param recvbuf Output data; should already be allocated.
 * @param count Length of sendbuf.
 * @param comm The communicator to allgather over.
 * @param algo Request a particular allgather algorithm.
 */
template <typename Backend, typename T>
void Allgather(const T* sendbuf, T* recvbuf, size_t count,
               typename Backend::comm_type& comm,
               typename Backend::allgather_algo_type algo =
               Backend::allgather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("allgather", comm, sendbuf, recvbuf,
                                         count);
  Backend::template Allgather<T>(sendbuf, recvbuf, count, comm, algo);
}

/**
 * Perform an in-place allgather.
 * @param recvbuf Input and output data; input will be overwritten.
 * @param count Length of data to send.
 * @param comm The communicator to allgather over.
 * @param algo Request a particular allgather algorithm.
 */
template <typename Backend, typename T>
void Allgather(T* recvbuf, size_t count,
               typename Backend::comm_type& comm,
               typename Backend::allgather_algo_type algo =
               Backend::allgather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("allgather", comm, recvbuf, count);
  Backend::template Allgather<T>(recvbuf, count, comm, algo);
}

/**
 * Non-blocking version of allgather.
 * This returns immediately (i.e. does only local operations) and starts the
 * allgather asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request indicates that
 * the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingAllgather(
  const T* sendbuf, T* recvbuf, size_t count,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::allgather_algo_type algo =
  Backend::allgather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-allgather", comm,
                                         sendbuf, recvbuf, count);
  Backend::template NonblockingAllgather<T>(sendbuf, recvbuf, count, comm, req, algo);
}

/** In-place version of NonblockingAllgather; same semantics apply. */
template <typename Backend, typename T>
void NonblockingAllgather(
  T* recvbuf, size_t count,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::allgather_algo_type algo =
  Backend::allgather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-allgather", comm,
                                         recvbuf, count);
  Backend::template NonblockingAllgather<T>(recvbuf, count, comm, req, algo);
}

// There are no in-place broadcast versions; it is always in-place.

/**
 * Perform a broadcast.
 * @param buf Data to send on root; received into on other processes.
 * @param count Length of buf.
 * @param root The root of the broadcast.
 * @param comm The communicator to broadcast over.
 * @param algo Request a particular broadcast algorithm.
 */
template <typename Backend, typename T>
void Bcast(T* buf,
           size_t count,
           int root,
           typename Backend::comm_type& comm,
           typename Backend::bcast_algo_type algo =
           Backend::bcast_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("bcast", comm, buf, count, root);
  Backend::template Bcast<T>(buf, count, root, comm, algo);
}

/**
 * Non-blocking version of Bcast.
 * This returns immediately (i.e. does only local operations) and starts the
 * broadcast asynchronously.
 * It is not safe to modify buf until the request indicates that the operation
 * has completed.
 */
template <typename Backend, typename T>
void NonblockingBcast(
  T* buf,
  size_t count,
  int root,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::bcast_algo_type algo =
  Backend::bcast_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-bcast", comm, buf,
                                         count, root);
  Backend::template NonblockingBcast<T>(buf,  count, root, comm, req, algo);
}

/**
 * All-to-all scatter/gather operation.
 * @param sendbuf The source data buffer.
 * @param recvbuf The destination data buffer.
 * @param count The per-rank data count.
 * @param comm The communicator for this all-to-all operation.
 * @param algo Request a particular all-to-all algorithm.
 */
template <typename Backend, typename T>
void Alltoall(
  const T* sendbuf, T* recvbuf, size_t count,
  typename Backend::comm_type& comm,
  typename Backend::alltoall_algo_type algo =
  Backend::alltoall_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("alltoall", comm, sendbuf, recvbuf,
                                         count);
  Backend::template Alltoall<T>(sendbuf, recvbuf, count, comm, algo);
}

/**
 * In-place all-to-all scatter/gather operation.
 * @param buffer The data buffer; overwritten on completion.
 * @param count The per-rank data count.
 * @param comm The communicator fo this all-to-all operation.
 * @param algo Request a particular all-to-all algorithm.
 */
template <typename Backend, typename T>
void Alltoall(
  T* buffer, size_t count, typename Backend::comm_type& comm,
  typename Backend::alltoall_algo_type algo =
  Backend::alltoall_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("alltoall", comm, buffer, count);
  Backend::template Alltoall<T>(buffer, count, comm, algo);
}

/**
 * Non-blocking version of Alltoall.
 * This returns immediately (i.e. does only local operations) and
 * starts the all-to-all asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request
 * indicates that the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingAlltoall(
  const T* sendbuf, T* recvbuf, size_t count,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::alltoall_algo_type algo =
  Backend::alltoall_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-alltoall", comm, sendbuf,
                                         recvbuf, count);
  Backend::template NonblockingAlltoall<T>(sendbuf, recvbuf, count,
                                           comm, req, algo);
}

/** In-place version of NonblockingAlltoall; same semantics apply. */
template <typename Backend, typename T>
void NonblockingAlltoall(
  T* buffer, size_t count, typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::alltoall_algo_type algo =
  Backend::alltoall_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-alltoall", comm, buffer,
                                         count);
  Backend::template NonblockingAlltoall<T>(buffer, count, comm, req, algo);
}

/**
 * Gather-to-one operation.
 * @param sendbuf The source data buffer.
 * @param recvbuf The destination data buffer.
 * @param count The per-rank data count.
 * @param root The root process to which data is gathered.
 * @param comm The communicator for this gather operation.
 * @param algo Request a particular gather algorithm.
 */
template <typename Backend, typename T>
void Gather(
  const T* sendbuf, T* recvbuf, size_t count, int root,
  typename Backend::comm_type& comm,
  typename Backend::gather_algo_type algo =
  Backend::gather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("gather", comm, sendbuf, recvbuf,
                                         count, root);
  Backend::template Gather<T>(sendbuf, recvbuf, count, root, comm, algo);
}

/**
 * In-place gather-to-one operation.
 * @param buffer The data buffer; overwritten on completion. For root
 *               processes, has size count*comm.size() and acts as
 *               recvbuf above. For nonroot processes, has size count
 *               and acts as sendbuf above.
 * @param count The per-rank data count.
 * @param root The root process to which data is gathered.
 * @param comm The communicator fo this gather operation.
 * @param algo Request a particular gather algorithm.
 */
template <typename Backend, typename T>
void Gather(
  T* buffer, size_t count, int root, typename Backend::comm_type& comm,
  typename Backend::gather_algo_type algo =
  Backend::gather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("gather", comm, buffer, count, root);
  Backend::template Gather<T>(buffer, count, root, comm, algo);
}

/**
 * Non-blocking version of Gather.
 * This returns immediately (i.e. does only local operations) and
 * starts the gather asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request
 * indicates that the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingGather(
  const T* sendbuf, T* recvbuf, size_t count, int root,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::gather_algo_type algo =
  Backend::gather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-gather", comm, sendbuf,
                                         recvbuf, count, root);
  Backend::template NonblockingGather<T>(sendbuf, recvbuf, count, root,
                                         comm, req, algo);
}

/** In-place version of NonblockingGather; same semantics apply. */
template <typename Backend, typename T>
void NonblockingGather(
  T* buffer, size_t count, int root, typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::gather_algo_type algo =
  Backend::gather_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-gather", comm, buffer,
                                         count, root);
  Backend::template NonblockingGather<T>(buffer, count, root, comm, req, algo);
}

/**
 * Scatter-to-all operation.
 * @param sendbuf The source data buffer.
 * @param recvbuf The destination data buffer.
 * @param count The per-rank data count.
 * @param root The root process from which data is scattered.
 * @param comm The communicator for this scatter operation.
 * @param algo Request a particular scatter algorithm.
 */
template <typename Backend, typename T>
void Scatter(
  const T* sendbuf, T* recvbuf, size_t count, int root,
  typename Backend::comm_type& comm,
  typename Backend::scatter_algo_type algo =
  Backend::scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("scatter", comm, sendbuf, recvbuf,
                                         count, root);
  Backend::template Scatter<T>(sendbuf, recvbuf, count, root, comm, algo);
}

/**
 * In-place scatter-to-all operation.
 * @param buffer The data buffer; overwritten on completion. For root
 *               processes, has size count*comm.size() and acts as
 *               sendbuf above. For nonroot processes, has size count
 *               and acts as recvbuf above.
 * @param count The per-rank data count.
 * @param root The root process from which data is scattered.
 * @param comm The communicator fo this scatter operation.
 * @param algo Request a particular scatter algorithm.
 */
template <typename Backend, typename T>
void Scatter(
  T* buffer, size_t count, int root, typename Backend::comm_type& comm,
  typename Backend::scatter_algo_type algo =
  Backend::scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("scatter", comm, buffer, count, root);
  Backend::template Scatter<T>(buffer, count, root, comm, algo);
}

/**
 * Non-blocking version of Scatter.
 * This returns immediately (i.e. does only local operations) and
 * starts the scatter asynchronously.
 * It is not safe to modify sendbuf or recvbuf until the request
 * indicates that the operation has completed.
 */
template <typename Backend, typename T>
void NonblockingScatter(
  const T* sendbuf, T* recvbuf, size_t count, int root,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::scatter_algo_type algo =
  Backend::scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-scatter", comm, sendbuf,
                                         recvbuf, count, root);
  Backend::template NonblockingScatter<T>(sendbuf, recvbuf, count, root,
                                         comm, req, algo);
}

/** In-place version of NonblockingScatter; same semantics apply. */
template <typename Backend, typename T>
void NonblockingScatter(
  T* buffer, size_t count, int root, typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::scatter_algo_type algo =
  Backend::scatter_algo_type::automatic) {
  internal::trace::record_op<Backend, T>("nonblocking-scatter", comm, buffer,
                                         count, root);
  Backend::template NonblockingScatter<T>(buffer, count, root, comm, req, algo);
}

/**
 * Send a point-to-point message.
 * @param sendbuf The data to send.
 * @param count Length of sendbuf.
 * @param dest Rank in comm to send to.
 * @param comm Communicator to send within.
 */
template <typename Backend, typename T>
void Send(const T* sendbuf, size_t count, int dest,
          typename Backend::comm_type& comm) {
  internal::trace::record_op<Backend, T>("send", comm, sendbuf, count, dest);
  Backend::template Send<T>(sendbuf, count, dest, comm);
}

/** Non-blocking version of Send. */
template <typename Backend, typename T>
void NonblockingSend(const T* sendbuf, size_t count, int dest,
                     typename Backend::comm_type& comm,
                     typename Backend::req_type& req) {
  internal::trace::record_op<Backend, T>("nonblocking-send", comm, sendbuf,
                                         count, dest);
  Backend::template NonblockingSend<T>(sendbuf, count, dest, comm, req);
}

/**
 * Receive a point-to-point message.
 * @param recvbuf Buffer to receive into.
 * @param count Length of recvbuf.
 * @param src Rank in comm to receive from.
 * @param comm Communicator to receive within.
 */
template <typename Backend, typename T>
void Recv(T* recvbuf, size_t count, int src,
          typename Backend::comm_type& comm) {
  internal::trace::record_op<Backend, T>("recv", comm, recvbuf, count, src);
  Backend::template Recv<T>(recvbuf, count, src, comm);
}

/** Non-blocking version of Recv. */
template <typename Backend, typename T>
void NonblockingRecv(T* recvbuf, size_t count, int src,
                     typename Backend::comm_type& comm,
                     typename Backend::req_type& req) {
  internal::trace::record_op<Backend, T>("nonblocking-recv", comm, recvbuf,
                                         count, src);
  Backend::template NonblockingRecv<T>(recvbuf, count, src, comm, req);
}

/**
 * Perform a simultaneous send and recv.
 * @param sendbuf The data to send.
 * @param send_count Length of sendbuf.
 * @param dest Rank in comm to send to.
 * @param recvbuf Buffer to receive into.
 * @param recv_count Length of recvbuf.
 * @param src Rank in comm to receive from.
 * @param comm Communicator to send/recv within.
 */
template <typename Backend, typename T>
void SendRecv(const T* sendbuf, size_t send_count, int dest,
              T* recvbuf, size_t recv_count, int src,
              typename Backend::comm_type& comm) {
  internal::trace::record_op<Backend, T>("sendrecv", comm, sendbuf, send_count,
                                         dest, recvbuf, recv_count, src);
  Backend::template SendRecv<T>(sendbuf, send_count, dest,
                                recvbuf, recv_count, src, comm);
}

template <typename Backend, typename T>
void NonblockingSendRecv(const T* sendbuf, size_t send_count, int dest,
                         T* recvbuf, size_t recv_count, int src,
                         typename Backend::comm_type& comm,
                         typename Backend::req_type& req) {
  internal::trace::record_op<Backend, T>("nonblocking-sendrecv", comm,
                                         sendbuf, send_count, dest,
                                         recvbuf, recv_count, src);
  Backend::template NonblockingSendRecv<T>(sendbuf, send_count, dest,
                                           recvbuf, recv_count, src,
                                           comm, req);
}

/**
 * Test whether req has completed or not, returning true if it has.
 */
template <typename Backend>
bool Test(typename Backend::req_type& req);
/** Wait until req has been completed. */
template <typename Backend>
void Wait(typename Backend::req_type& req);

namespace ext {

#ifdef AL_HAS_MPI_CUDA_RMA
/**
 * Attach a remote buffer to local memory space for RMA.
 * @param local_buf Local buffer attached by remote rank.
 * @param peer Rank in comm to attach buffers with.
 * @param comm Communicator to attach buffers.
 * @return Local address the remote buffer is attached at.
 */
template <typename Backend, typename T>
T *AttachRemoteBuffer(T *local_buf, int peer,
                      typename Backend::comm_type& comm) {
  return Backend::template AttachRemoteBuffer<T>(local_buf, peer, comm);
}

/**
 * Detach an attached buffer from local memory space.
 * @param remote_buf Buffer previously attached.
 * @param peer Rank in comm the buffer is attached with.
 * @param comm Communicator the buffer is attached with.
 */
template <typename Backend, typename T>
void DetachRemoteBuffer(T *remote_buf, int peer,
                        typename Backend::comm_type& comm) {
  Backend::template DetachRemoteBuffer<T>(remote_buf, peer, comm);
}

/**
 * Send a notification message.
 * @param peer Rank in comm to send a notification to.
 * @param comm Communicator to send a notification within.
 */
template <typename Backend>
void Notify(int peer, typename Backend::comm_type& comm) {
  Backend::Notify(peer, comm);
}

/**
 * Wait a notification message.
 * @param peer Rank in comm to wait a notification from.
 * @param comm Communicator to wait a notification within.
 */
template <typename Backend>
void Wait(int peer, typename Backend::comm_type& comm) {
  Backend::Wait(peer, comm);
}

/**
 * Exchange a notification message.
 * @param peer Rank in comm to exchange a notification with.
 * @param comm Communicator to exchange a notification within.
 */
template <typename Backend>
void Sync(int peer, typename Backend::comm_type& comm) {
  Backend::Sync(peer, comm);
}

/**
 * Exchange notification messages with multiple ranks.
 * @param peers Ranks in comm to exchange a notification with.
 * @param num_peers Number of ranks in peers.
 * @param comm Communicator to exchange notifications within.
 */
template <typename Backend>
void Sync(const int *peers, int num_peers,
          typename Backend::comm_type& comm) {
  Backend::Sync(peers, num_peers, comm);
}

/**
 * Put a point-to-point message.
 * @param srcbuf The data to put.
 * @param count Length of srcbuf.
 * @param dest Rank in comm to put to.
 * @param destbuf Buffer to put to.
 * @param comm Communicator to put within.
 */
template <typename Backend, typename T>
void Put(const T* srcbuf, int dest, T *destbuf,
         size_t count, typename Backend::comm_type& comm) {
  Backend::template Put<T>(srcbuf, dest, destbuf, count, comm);
}
#endif // AL_HAS_MPI_CUDA_RMA

} // namespace ext

}  // namespace Al

#include "mpi_impl.hpp"

#ifdef AL_HAS_NCCL
#include "nccl_impl.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "mpi_cuda_impl.hpp"
#endif
#ifdef AL_HAS_CUDA_AWARE_MPI
#include "cuda_aware_mpi_impl.hpp"
#endif
