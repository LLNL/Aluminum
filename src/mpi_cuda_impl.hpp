#pragma once

#include "Al.hpp"
#include "mpi_cuda/allreduce.hpp"

namespace Al {

enum class MPICUDAAllreduceAlgorithm {
  automatic,
  ring,
  bi_ring,
  host_transfer
};

inline std::string allreduce_name(MPICUDAAllreduceAlgorithm algo) {
  switch (algo) {
  case MPICUDAAllreduceAlgorithm::automatic:
    return "automatic";
  case MPICUDAAllreduceAlgorithm::ring:
    return "ring";
  case MPICUDAAllreduceAlgorithm::bi_ring:
    return "bi-ring";
  case MPICUDAAllreduceAlgorithm::host_transfer:
    return "host-transfer";
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
  using algo_type = MPICUDAAllreduceAlgorithm;
  using comm_type = internal::mpi_cuda::MPICUDACommunicator;
  using req_type = std::shared_ptr<internal::mpi_cuda::MPICUDARequest>;
  static constexpr std::nullptr_t null_req = nullptr;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
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
    case MPICUDAAllreduceAlgorithm::host_transfer:
      do_host_transfer_allreduce(sendbuf, recvbuf, count, op, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }
  template <typename T>
  static void Allreduce(T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    Allreduce(recvbuf, recvbuf, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingAllreduce(
      const T* sendbuf, T* recvbuf, size_t count,
      ReductionOperator op,
      comm_type& comm,
      req_type& req,
      algo_type algo) {
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
    case MPICUDAAllreduceAlgorithm::host_transfer:
      do_nonblocking_host_transfer_allreduce(
        sendbuf, recvbuf, count, op, comm, req, internal_stream);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
    if (algo != MPICUDAAllreduceAlgorithm::host_transfer) {
      // Completion for host-transfer handled inside
      // do_nonblocking_host_transfer_allreduce.
      setup_completion_event(internal_stream, comm, req);
    }
  }

  template <typename T>
  static void NonblockingAllreduce(
      T* recvbuf, size_t count,
      ReductionOperator op, comm_type& comm,
      req_type& req,
      algo_type algo) {
    NonblockingAllreduce(recvbuf, recvbuf, count, op, comm,
                         req, algo);
  }

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

  /** Run a blocking host-transfer allreduce. */
  template <typename T>
  static void do_host_transfer_allreduce(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
    comm_type& comm) {
    // Get pinned host memory.
    T* host_mem = internal::get_pinned_memory<T>(count);
    internal::mpi_cuda::host_transfer_allreduce(
      sendbuf, recvbuf, host_mem, count, op, comm, comm.get_stream());
    // We can only free the memory after the allreduce has completed, but don't
    // want to block the user's stream to do so.
    cudaStream_t internal_stream = internal::cuda::get_internal_stream();
    sync_internal_stream_with_comm(internal_stream, comm);
    AL_CHECK_CUDA(cudaStreamAddCallback(
                    internal_stream,
                    internal::mpi_cuda::host_transfer_allreduce_free_mem<T>,
                    (void*) host_mem, 0));
  }

  /** Run a non-blocking host-transfer allreduce. */
  template <typename T>
  static void do_nonblocking_host_transfer_allreduce(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
    comm_type& comm, req_type& req, cudaStream_t internal_stream) {
    // Get pinned host memory.
    T* host_mem = internal::get_pinned_memory<T>(count);
    internal::mpi_cuda::host_transfer_allreduce(
      sendbuf, recvbuf, host_mem, count, op, comm, internal_stream);
    // Set up the completion event before freeing memory.
    setup_completion_event(internal_stream, comm, req);
    // Now set up the callback to free memory.
    AL_CHECK_CUDA(cudaStreamAddCallback(
                    internal_stream,
                    internal::mpi_cuda::host_transfer_allreduce_free_mem<T>,
                    (void*) host_mem, 0));
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
