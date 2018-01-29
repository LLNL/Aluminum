/**
 * Various optimized allreduce implementations.
 */

#pragma once

#include <iostream>
#include <exception>
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

#include "tuning_params.hpp"

#include "nccl.h"
#include "common.h"

namespace allreduces {

/**
 * Base allreduce exception class.
 */
class allreduce_exception : public std::exception {
 public:
  allreduce_exception(const std::string m, const std::string f, const int l) :
    msg(m), file(f), line(l) {
    err = file + ":" + std::to_string(line) + " - " + msg;
  }
  const char* what() const noexcept override {
    return err.c_str();
  }
private:
  /** Exception message. */
  const std::string msg;
  /** File exception occurred in. */
  const std::string file;
  /** Line exception occurred at. */
  const int line;
  /** Constructed error message. */
  std::string err;
};
#define throw_allreduce_exception(s) throw allreduce_exception(s, __FILE__, __LINE__)

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
  ~MPICommunicator() override {
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

/** Predefined reduction operations. */
enum class ReductionOperator {
  sum, prod, min, max
};

/** Request handle for non-blocking allreduces. */
using AllreduceRequest = int;  // TODO: This is a placeholder.

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
  mpi_pe_ring
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
  default:
    return "unknown";
  }
}

/**
 * Initialize the allreduce library.
 * This must be called before any other calls to the library. It is safe to
 * call this multiple times.
 */
void Initialize(int& argc, char**& argv);
/**
 * Clean up the allreduce library.
 * Do not make any further calls to the library after calling this.
 */
void Finalize();
/** Return true if the library has been initialized. */
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
template <typename T>
void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
               ReductionOperator op, Communicator& comm,
               AllreduceAlgorithm algo = AllreduceAlgorithm::automatic);
/**
 * Perform an in-place allreduce.
 * @param recvbuf Input and output data; input will be overwritten.
 * @param count Length of sendbuf and recvbuf.
 * @param op The reduction operation to perform.
 * @param comm The communicator to reduce over.
 * @param algo Request a particular allreduce algorithm.
 */
template <typename T>
void Allreduce(T* recvbuf, size_t count,
               ReductionOperator op, Communicator& comm,
               AllreduceAlgorithm algo = AllreduceAlgorithm::automatic);

/**
 * Non-blocking version of Allreduce.
 * This returns immediately (i.e. does only local operations) and starts the
 * allreduce asynchronously. The request object is set to an opaque reference
 * for the allreduce, and can be checked using Test and Wait.
 * It is not safe to modify sendbuf or recvbuf until the request indicates that
 * the operation has completed.
 */
template <typename T>
void NonblockingAllreduce(
  const T* sendbuf, T* recvbuf, size_t count,
  ReductionOperator op, Communicator& comm,
  AllreduceRequest& req,
  AllreduceAlgorithm algo = AllreduceAlgorithm::automatic);
/** In-place version of NonblockingAllreduce; same semantics apply. */
template <typename T>
void NonblockingAllreduce(
  T* recvbuf, size_t count,
  ReductionOperator op, Communicator& comm,
  AllreduceRequest& req,
  AllreduceAlgorithm algo = AllreduceAlgorithm::automatic);

/**
 * Test whether req has completed or not, returning true if it has.
 */
bool Test(AllreduceRequest req);
/** Wait until req has been completed. */
void Wait(AllreduceRequest req);

/**
 * Internal implementations of allreduce.
 * Generic code for all allreduce implementations is in here.
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

/** Return a free allreduce request for use. */
AllreduceRequest get_free_request();
/** Special marker for null requests. */
static const AllreduceRequest NULL_REQUEST = 0;

/**
 * Represents the state and algorithm for an asynchronous allreduce.
 * A non-blocking allreduce should create one of these and enqueue it for
 * execution by the progress thread. Specific implementations can override
 * as needed.
 * An algorithm should be broken up into steps which execute some small,
 * discrete operation. Steps from different allreduces may be interleaved.
 * Note that the memory pool is not thread-safe, so memory from it should be
 * pre-allocated before enqueueing.
 */
class AllreduceState {
 public:
  /** Create with an associated allreduce request. */
  AllreduceState(AllreduceRequest req_) : req(req_) {}
  virtual ~AllreduceState() {}
  /**
   * Start one step of the allreduce algorithm.
   * Return true if the allreduce has completed, false if it has more steps
   * remaining.
   */
  virtual bool step() = 0;
  /** Return the associated request. */
  AllreduceRequest get_req() const { return req; }
 private:
  AllreduceRequest req;
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
  void enqueue(AllreduceState* state);
  /**
   * Check whether a request has completed.
   * If the request is completed, it is removed.
   * This does not block and may spuriously return false.
   */
  bool is_complete(AllreduceRequest& req);
  /**
   * Wait until a request has completed, then remove it.
   * This will block the calling thread.
   */
  void wait_for_completion(AllreduceRequest& req);
 private:
  /** The actual thread of execution. */
  std::thread thread;
  /** Atomic flag indicating the progress engine should stop; true to stop. */
  std::atomic<bool> stop_flag;
  /**
   * The list of requests that have been enqueued to the progress engine, but
   * that it has not yet begun to process. Threads may add states for
   * asynchronous execution, and the progress engine will dequeue them when it
   * begins to run them.
   * This is protected by the enqueue_mutex.
   */
  std::queue<AllreduceState*> enqueued_reqs;
  /** Protects enqueued_reqs. */
  std::mutex enqueue_mutex;
#if ALLREDUCE_PE_SLEEPS
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
  std::list<AllreduceState*> in_progress_reqs;
  /**
   * Requests that have been completed.
   * The request is added by the progress engine once it has been completed.
   * States should be deallocated by whatever removes them from this.
   * This is protected by the completed_mutex.
   */
  std::unordered_map<AllreduceRequest, AllreduceState*> completed_reqs;
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

/** MPI-based allreduce implementations. */
namespace mpi {

/** MPI initialization. */
void init(int& argc, char**& argv);
/** MPI finalization. */
void finalize();

/** Just call MPI_Allreduce directly. */
template <typename T>
void passthrough_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                           ReductionOperator op, Communicator& comm);
/** Just call MPI_Iallreduce directly. */
template <typename T>
void nb_passthrough_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                              ReductionOperator op, Communicator& comm,
                              AllreduceRequest& req);
/** Use a recursive-doubling algorithm to perform the allreduce. */
template <typename T>
void recursive_doubling_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                  ReductionOperator op, Communicator& comm);
/** Non-blocking recursive-doubling allreduce. */
template <typename T>
void nb_recursive_doubling_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                     ReductionOperator op, Communicator& comm,
                                     AllreduceRequest& req);
/** Use a ring-based reduce-scatter then allgather to perform the allreduce. */
template <typename T>
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, Communicator& comm);
/** Non-blocking ring allreduce. */
template <typename T>
void nb_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, Communicator& comm,
                       AllreduceRequest& req);
/**
 * Use Rabenseifner's algorithm (recursive halving/doubling) to perform the
 * allreduce.
 */
template <typename T>
void rabenseifner_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                            ReductionOperator op, Communicator& comm);
/** Non-blocking Rabenseifner allreduce. */
template <typename T>
void nb_rabenseifner_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                               ReductionOperator op, Communicator& comm,
                               AllreduceRequest& req);
/**
 * Use a pairwise-exchange reduce-scatter and ring allgather to perform the
 * allreduce.
 */
template <typename T>
void pe_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, Communicator& comm);

}  // namespace mpi

}  // namespace internal


/// We assume NCCL version 2.0 or higher for allreduce to work
class NCCLCommunicator : public MPICommunicator {
 public:

  /// NCCL communicator MUST operate in conjunction with an MPI_Comm 
  /// Default constructor; use MPI_COMM_WORLD
  NCCLCommunicator() : NCCLCommunicator(MPI_COMM_WORLD) {}

  /// NCCLCommunicator with an MPI communicator given
  NCCLCommunicator(MPI_Comm comm_) : MPICommunicator(comm_) {

    mpicomm = get_comm();

    m_nccl_used = true;

    /// Set up GPU-related informatiton
    gpu_setup();

    /// NCCL set up here
    nccl_setup();

    test_code();
  }

  ~NCCLCommunicator() override {
    /// NCCL destroy here
    nccl_destroy();
  }

  void test_code(){

  }

  /// It is assumed that both sendbuf and recvbuf are in device memory
  /// for NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  void Allreduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
               ReductionOperator op) {
               //ReductionOperator op, Communicator& comm) {

    if(count == 0) return;

    MPI_Comm comm_ = get_comm();

    int num_gpus_assigned = m_gpus.size();

    /// Convert type T to corresponding NCCL data type.
    switch(sizeof(nccl_type)) {
    case 8:
      nccl_type = ncclDouble;
      break;
    case 4:
      nccl_type = ncclFloat;
      break;
    case 2:
      nccl_type = ncclHalf;
      break;
    default:
      std::cerr << "NCCLCommunicator: rank " << rank() << ": invalid data type for NCCL\n";
      MPI_Abort(comm_, -4);
    }

    /// Convert ReductionOperator to corresponding NCCL reduction operation
    ncclRedOp_t nccl_redop;
    switch(op) {
    case ReductionOperator::sum:
      nccl_redop = ncclSum;
      break;
    case ReductionOperator::prod:
      nccl_redop = ncclProd;
      break;
    case ReductionOperator::min:
      nccl_redop = ncclMin;
      break;
    case ReductionOperator::max:
      nccl_redop = ncclMax;
      break;
    default:
      std::cerr << "NCCLCommunicator: rank " << rank() << ": invalid NCCL reduction operator\n";
      MPI_Abort(comm_, -5);
    }

    if(num_gpus_assigned > 1) ncclGroupStart();
    for(int i = 0; i < num_gpus_assigned; ++i) {
      CUDACHECK(cudaSetDevice(m_gpus[i]));
      NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, nccl_type, nccl_redop, m_nccl_comm[i], m_streams[i]));
    }
    if(num_gpus_assigned > 1) ncclGroupEnd();

  }


  bool is_nccl_used() { return m_nccl_used; }

  void gpu_setup() {

    const int rank_in_node = local_rank();
    const int procs_per_node = local_size();

    /// Determine number of visible GPUs on the current node
    CUDACHECK(cudaGetDeviceCount(&m_num_visible_gpus));
    if(m_num_visible_gpus < 1) {
      std::cerr << "NCCLCommunicator: rank " << rank() << ": has no GPUs found on the node\n";
      MPI_Abort(mpicomm, -1);
    }
    if(m_num_visible_gpus < procs_per_node) {
      std::cerr << "NCCLCommunicator: rank " << rank() << ": has not enough GPUs available for given ranks\n";
      MPI_Abort(mpicomm, -2);
    }
    else{
      /// The number of GPUs on this node is greater than or equal to that of ranks assigned to this node;
      /// ensure that the right number of GPUs are used
      m_num_visible_gpus = procs_per_node;
    }

    // Assign GPUs to process
    int gpu_start, gpu_end;
    
    const int gpus_per_proc = m_num_visible_gpus / procs_per_node;
    const int num_leftover_gpus = m_num_visible_gpus % procs_per_node;
    gpu_start = rank_in_node * gpus_per_proc;
    gpu_end = (rank_in_node + 1) * gpus_per_proc;
    if(rank_in_node < num_leftover_gpus) {
      gpu_start += rank_in_node;
      gpu_end += rank_in_node + 1;
    }
    else {
      gpu_start += num_leftover_gpus;
      gpu_end += num_leftover_gpus;
    }

    // Construct GPU objects
    for(int gpu = gpu_start; gpu < gpu_end; ++gpu) {
      CUDACHECK(cudaSetDevice(gpu));
      m_gpus.push_back(gpu);
      m_streams.push_back(nullptr);

      CUDACHECK(cudaStreamCreate(&m_streams.back()));
    }

    // Get number of GPUs for current MPI rank
    m_num_gpus = m_gpus.size();
  }


  void nccl_setup() {

    if(m_num_gpus != 1){
      std::cerr << "NCCLCommunicator: rank " << rank() << ": the number of GPUs assigned to process is " << m_num_gpus << "; should be 1\n";
      MPI_Abort(mpicomm, -3);
    }

    /// Create nccl communicators
    int num_gpus_assigned = m_num_gpus;
    m_nccl_comm.resize(num_gpus_assigned);

    int nProcs = size();
    int myid = rank();
    int total_num_comms = nProcs*num_gpus_assigned;

    ncclUniqueId ncclId;
    if (myid == 0) {
      NCCLCHECK(ncclGetUniqueId(&ncclId));
    }

    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, mpicomm);

    if (nProcs == 1) {
      int gpuArray = 0;
      NCCLCHECK(ncclCommInitAll(&(m_nccl_comm[0]), 1, &gpuArray));
    }
    else {
      if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupStart());
      for(int i=0; i<num_gpus_assigned; i++){
        CUDACHECK(cudaSetDevice(m_gpus[i]));
        NCCLCHECK(ncclCommInitRank(&(m_nccl_comm[i]), total_num_comms, ncclId, num_gpus_assigned*myid+i));
      }
      if(num_gpus_assigned > 1) NCCLCHECK(ncclGroupEnd());
    }
  } // nccl_setup

  void nccl_destroy() {
    int num_gpus_assigned = m_gpus.size();
    for(int i=0; i<num_gpus_assigned; i++){
      ncclCommDestroy(m_nccl_comm[i]);
    }
  }


 private:

  MPI_Comm mpicomm;

  /** List of GPU related variables. */
  /// List of GPUs to be used
  std::vector<int> m_gpus;
  /// List of CUDA streams
  std::vector<cudaStream_t> m_streams;
  /// Number of GPUs allocated to the current rank
  int m_num_gpus;
  /// Number of visible GPUs on this compute node
  int m_num_visible_gpus;


  /** List of NCCL 2 related variables. */
  /// NOTE: It is assumed that ONLY ONE GPU is allocated to one MPI rank
  bool m_nccl_used;
  std::vector<ncclComm_t> m_nccl_comm;
};
}  // namespace allreduces

#include "allreduce_impl.hpp"
#include "allreduce_mempool.hpp"
#include "allreduce_mpi_impl.hpp"


