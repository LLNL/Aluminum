/**
 * Various optimized collective implementations.
 */

#pragma once

#include <iostream>
#include <limits>
#include <functional>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <algorithm>
#include <mpi.h>

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
};

/**
 * Communicator for MPI-based collectives.
 */
class MPICommunicator : public Communicator {
 public:
  /** Default constructor; use MPI_COMM_WORLD. */
  MPICommunicator() : MPICommunicator(MPI_COMM_WORLD) {}
  /** Use a particular MPI communicator. */
  MPICommunicator(MPI_Comm comm_) : Communicator() {
    // Duplicate the communicator to avoid interference.
    MPI_Comm_dup(comm_, &comm);
    MPI_Comm_rank(comm, &rank_in_comm);
    MPI_Comm_size(comm, &size_of_comm);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &rank_in_local_comm);
    MPI_Comm_size(local_comm, &size_of_local_comm);
  }
  virtual ~MPICommunicator() override {
    // TODO: Fix; can't do this after finalization.
    //MPI_Comm_free(&comm);
  }
  Communicator* copy() const override { return new MPICommunicator(comm); }
  int rank() const override { return rank_in_comm; }
  int size() const override { return size_of_comm; }
  MPI_Comm get_comm() const { return comm; }
  int local_rank() const { return rank_in_local_comm; }
  int local_size() const { return size_of_local_comm; }
  MPI_Comm get_local_comm() const { return local_comm; }
  int get_free_tag() { return free_tag++; }
  
 private:
  /** Associated MPI communicator. */
  MPI_Comm comm;
  /** Rank in comm. */
  int rank_in_comm;
  /** Size of comm. */
  int size_of_comm;
  /** Communicator for the local node. */
  MPI_Comm local_comm;
  /** Rank in the local communicator. */
  int rank_in_local_comm;
  /** Size of the local communicator. */
  int size_of_local_comm;
  /** Free tag for communication. */
  int free_tag = 1;
};

/**
 * Supported allreduce algorithms.
 * This is used for requesting a particular algorithm. Use automatic to let the
 * library select for you.
 */
enum class AllreduceAlgorithm {
  automatic,
  mpi_passthrough,
  mpi_recursive_doubling,
  mpi_ring,
  mpi_rabenseifner,
  mpi_pe_ring,
  mpi_biring
};

inline std::string allreduce_name(AllreduceAlgorithm algo) {
  switch (algo) {
  case AllreduceAlgorithm::automatic:
    return "automatic";
  case AllreduceAlgorithm::mpi_passthrough:
    return "MPI passthrough";
  case AllreduceAlgorithm::mpi_recursive_doubling:
    return "MPI recursive doubling";
  case AllreduceAlgorithm::mpi_ring:
    return "MPI ring";
  case AllreduceAlgorithm::mpi_rabenseifner:
    return "MPI Rabenseifner";
  case AllreduceAlgorithm::mpi_pe_ring:
    return "MPI PE/ring";
  case AllreduceAlgorithm::mpi_biring:
    return "MPI biring";
  default:
    return "unknown";
  }
}

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

/**
 * Internal implementations.
 * Generic code for all collective implementations is in here.
 * Implementation-specific code is in separate namespaces inside internal.
 */
namespace internal {

// Would be nice to replace this with a C++14 variable template...
/** Indicator that an in-place alrleduce is requested. */
template <typename T>
inline T* IN_PLACE() { return (T*) (-1); }

// This interfaces to a simple memory pool.
// WARNING: THIS IS NOT THREAD SAFE.
/** Get memory of type T with count elements. */
template <typename T>
T* get_memory(size_t count);
/** Release memory that you got with get_memory. */
template <typename T>
void release_memory(T* mem);

/** Request handle for non-blocking operations.. */
using AlRequest = int;  // TODO: This is a placeholder.

/** Return a free request for use. */
AlRequest get_free_request();
/** Special marker for null requests. */
static const AlRequest NULL_REQUEST = 0;

/**
 * Represents the state and algorithm for an asynchronous operation.
 * A non-blocking operation should create one of these and enqueue it for
 * execution by the progress thread. Specific implementations can override
 * as needed.
 * An algorithm should be broken up into steps which execute some small,
 * discrete operation. Steps from different operations may be interleaved.
 * Note that the memory pool is not thread-safe, so memory from it should be
 * pre-allocated before enqueueing.
 */
class AlState {
 public:
  /** Create with an associated request. */
  AlState(AlRequest req_) : req(req_) {}
  virtual ~AlState() {}
  /**
   * Start one step of the algorithm.
   * Return true if the operation has completed, false if it has more steps
   * remaining.
   */
  virtual bool step() = 0;
  /** Return the associated request. */
  AlRequest get_req() const { return req; }
 private:
  AlRequest req;
};

/**
 * Encapsulates the asynchronous progress engine.
 * Note this is intended to be used from only one thread (in addition to the
 * progress thread) and is not optimized (or tested) for other cases.
 */
class ProgressEngine {
 public:
  ProgressEngine();
  ProgressEngine(const ProgressEngine&) = delete;
  ProgressEngine& operator=(const ProgressEngine) = delete;
  ~ProgressEngine();
  /** Start the progress engine. */
  void run();
  /** Stop the progress engine. */
  void stop();
  /** Enqueue state for asynchronous execution. */
  void enqueue(AlState* state);
  /**
   * Check whether a request has completed.
   * If the request is completed, it is removed.
   * This does not block and may spuriously return false.
   */
  bool is_complete(AlRequest& req);
  /**
   * Wait until a request has completed, then remove it.
   * This will block the calling thread.
   */
  void wait_for_completion(AlRequest& req);
 private:
  /** The actual thread of execution. */
  std::thread thread;
  /** Atomic flag indicating the progress engine should stop; true to stop. */
  std::atomic<bool> stop_flag;
  /** For startup_cv. */
  std::mutex startup_mutex;
  /** Used to signal to the main thread that the progress engine has started. */
  std::condition_variable startup_cv;
  /** Atomic flag indicating that the progress engine has completed startup. */
  std::atomic<bool> started_flag;
  /**
   * The list of requests that have been enqueued to the progress engine, but
   * that it has not yet begun to process. Threads may add states for
   * asynchronous execution, and the progress engine will dequeue them when it
   * begins to run them.
   * This is protected by the enqueue_mutex.
   */
  std::queue<AlState*> enqueued_reqs;
  /** Protects enqueued_reqs. */
  std::mutex enqueue_mutex;
#if AL_PE_SLEEPS
  /**
   * The progress engine will sleep on this when there is otherwise no work to
   * do.
   */
  std::condition_variable enqueue_cv;
#endif
  /**
   * Requests the progress engine is currently processing.
   * This should be accessed only by the progress engine (it is not protected).
   */
  std::list<AlState*> in_progress_reqs;
  /**
   * Requests that have been completed.
   * The request is added by the progress engine once it has been completed.
   * States should be deallocated by whatever removes them from this.
   * This is protected by the completed_mutex.
   */
  std::unordered_map<AlRequest, AlState*> completed_reqs;
  /** Protects completed_reqs. */
  std::mutex completed_mutex;
  /** Used to notify any thread waiting for completion. */
  std::condition_variable completion_cv;
  /**
   * World communicator.
   * Note: This means we require MPI, which may be something to change later,
   * but it simplifies things.
   */
  MPICommunicator* world_comm;
  /**
   * Bind the progress engine to a core.
   * This binds to the last core in the NUMA node the process is in.
   * If there are multiple ranks per NUMA node, they get the last-1, etc. core.
   */
  void bind();
  /** This is the main progress engine loop. */
  void engine();
};

/** Return a pointer to the progress engine. */
ProgressEngine* get_progress_engine();

}  // namespace internal
}  // namespace Al

#include "mempool.hpp"
#include "mpi_impl.hpp"

#ifdef AL_HAS_NCCL
#include "nccl_impl.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "mpi_cuda_impl.hpp"
#endif
