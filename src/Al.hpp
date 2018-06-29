/**
 * Various optimized collective implementations.
 */

#pragma once

#include <iostream>
#include <mpi.h>

#include "Al_config.hpp"
#include "base.hpp"
#include "tuning_params.hpp"

namespace Al {

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
               typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
               typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
            typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
            typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template Reduce<T>( recvbuf, count, op, root, comm, algo);
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
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template NonblockingReduce<T>(recvbuf, count, op, root, comm, req, algo);
}

/**
 * Perform a reduce-scatter.
 * @param sendbuf Input data.
 * @param recvbuf Output data; should already be allocated.
 * @param count Length of recvbuf.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce/scatter over.
 * @param algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(const T* sendbuf, T* recvbuf, size_t *count,
            ReductionOperator op, typename Backend::comm_type& comm,
            typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template Reduce_scatter<T>(sendbuf, recvbuf, count, op, comm, algo);
}

/**
 * Perform an in-place reduce-scatter.
 * @param recvbuf Input and output data; input will be overwritten.
 * @param count Length of data to be received.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce/scatter over.
 * @param algo Request a particular reduce-scatter algorithm.
 */
template <typename Backend, typename T>
void Reduce_scatter(T* recvbuf, size_t *count,
            ReductionOperator op, typename Backend::comm_type& comm,
            typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template Reduce_scatter<T>(recvbuf, count, op, comm, algo);
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
  const T* sendbuf, T* recvbuf, size_t *count,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template NonblockingReduce_scatter<T>(sendbuf, recvbuf, count, op, comm, req, algo);
}

/** In-place version of NonblockingReduce_scatter; same semantics apply. */
template <typename Backend, typename T>
void NonblockingReduce_scatter(
  T* recvbuf, size_t *count,
  ReductionOperator op,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template NonblockingReduce_scatter<T>(recvbuf, count, op, comm, req, algo);
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
               typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
               typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template NonblockingAllgather<T>(sendbuf, recvbuf, count, comm, req, algo);
}

/** In-place version of NonblockingAllgather; same semantics apply. */
template <typename Backend, typename T>
void NonblockingAllgather(
  T* recvbuf, size_t count,
  typename Backend::comm_type& comm,
  typename Backend::req_type& req,
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
           typename Backend::algo_type algo = Backend::algo_type::automatic) {
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
  typename Backend::algo_type algo = Backend::algo_type::automatic) {
  Backend::template NonblockingBcast<T>(buf,  count, root, comm, req, algo);
}

/**
 * Test whether req has completed or not, returning true if it has.
 */
template <typename Backend>
bool Test(typename Backend::req_type& req);
/** Wait until req has been completed. */
template <typename Backend>
void Wait(typename Backend::req_type& req);

}  // namespace Al

#include "mpi_impl.hpp"

#ifdef AL_HAS_NCCL
#include "nccl_impl.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "mpi_cuda_impl.hpp"
#endif
