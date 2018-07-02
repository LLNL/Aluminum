#pragma once

#include <nccl.h>
#include "Al.hpp"

namespace Al {

enum class NCCLCollectiveAlgorithm {
  automatic
};

inline std::string allreduce_name(NCCLCollectiveAlgorithm algo) {
  switch (algo) {
  case NCCLCollectiveAlgorithm::automatic:
    return "automatic";
  default:
    return "unknown";
  }
}

/**
 * Communicator for NCCL-based allreduces.
 * This requires NCCL version 2.0 or higher.
 * This currently requires there to be only one GPU per MPI rank.
 */
class NCCLCommunicator : public MPICommunicator {
 public:
  /**
   * Initialize a NCCL communicator.
   * @param comm_ An MPI_Comm representing the nodes to be in the communicator.
   * @param gpus The GPUs this communicator is (locally) managing.
   */
  NCCLCommunicator(MPI_Comm comm_ = MPI_COMM_WORLD,
                   std::vector<int> gpus = std::vector<int>());
  //~NCCLCommunicator();
  ~NCCLCommunicator() override;
  Communicator* copy() const override {
    return new NCCLCommunicator(get_comm(), m_gpus);
  }

  /** Synchronize the internal stream for each managed GPU. */
  void synchronize();

  /** Return the default stream for this communicator. */
  cudaStream_t get_default_stream();

  /**
   * Perform an allreduce on data in device memory.
   * @param sendbuf Input data.
   * @param recvbuf Output data; if same as sendbuf, an in-place allreduce is done.
   * @param count Number of elements in sendbuf/recvbuf.
   * @param nccl_type Type of data being reduced.
   * @param nccl_redop The reduction operation to perform.
   * @param default_stream CUDA stream to associate with the operation.
   */
  void Allreduce(const void* sendbuf, void* recvbuf, size_t count,
                 ncclDataType_t nccl_type, ncclRedOp_t nccl_redop,
                 cudaStream_t default_stream); 

  /**
   * Perform a reduce on data in device memory.
   * @param sendbuf Input data.
   * @param recvbuf Output data; if same as sendbuf, an in-place reduce is done.
   * @param count Number of elements in sendbuf/recvbuf.
   * @param nccl_type Type of data being reduced.
   * @param nccl_redop The reduction operation to perform.
   * @param root Rank to receive the final result.
   * @param default_stream CUDA stream to associate with the operation.
   */
  void Reduce(const void* sendbuf, void* recvbuf, size_t count,
              ncclDataType_t nccl_type, ncclRedOp_t nccl_redop, int root,
              cudaStream_t default_stream); 

  /**
   * Perform a broadcast on data in device memory.
   * @param buf Data to send on the root; receive buffer on other ranks.
   * @param count Number of elements in buf.
   * @param nccl_type Type of data being broadcast.
   * @param root Rank initiating the broadcast.
   * @param default_stream CUDA stream to associate with the operation.
   */
  void Bcast(void* buf, size_t count, ncclDataType_t nccl_type, int root, 
             cudaStream_t default_stream); 

  /**
   * Perform an allgather on data in device memory.
   * @param sendbuf Input data; if this is an offset into recvbuf, an in-place
   * operation is performed.
   * @param recvbuf Output data; of size count*ranks.
   * @param send_count Number of elements to send.
   * @param nccl_type Type of data being gathered.
   * @param default_stream CUDA stream to associate with the operation.
   */
  void Allgather(const void* sendbuf, void* recvbuf, size_t send_count,
                 ncclDataType_t nccl_type, cudaStream_t default_stream);

  /**
   * Perform a reduce-scatter on data in device memory.
   * @param sendbuf Input data; of size recv_count*ranks.
   * @param recvbuf Output data; if this is an offset into sendbuf, an in-place
   * operation is performed.
   * @param recv_count Number of elements to receive.
   * @param nccl_type Type of data being gathered.
   * @param nccl_redop The reduction operation to perform.
   * @param default_stream CUDA stream to associate with the operation.
   */
  void Reduce_scatter(const void* sendbuf, void* recvbuf, size_t recv_count,
                      ncclDataType_t nccl_type, ncclRedOp_t nccl_redop,
                      cudaStream_t default_stream);

 protected:
  /** Initialize GPU information. */
  void gpu_setup();
  /** Initialize the internal NCCL communicator. */
  void nccl_setup();
  /** Clean up NCCL. */
  void nccl_destroy();

 private:
  /** List of GPUs associated with this communicator. */
  std::vector<int> m_gpus;
  /** List of streams, one for each GPU. */
  std::vector<cudaStream_t> m_streams;
  /** GPUs allocated to this rank. */
  int m_num_gpus;
  /** NCCL communicator. */
  ncclComm_t m_nccl_comm;
};

namespace internal {
namespace nccl {
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
// TODO half-precision.
//template <> inline ncclDataType_t TypeMap<??>() { return ncclHalf; }
template <> inline ncclDataType_t TypeMap<float>() { return ncclFloat; }
template <> inline ncclDataType_t TypeMap<double>() { return ncclDouble; }
}  // namespace nccl
}  // namespace internal

/** Backend implementing NCCL communication. */
class NCCLBackend {
 public:
  using algo_type = NCCLCollectiveAlgorithm;
  using comm_type = NCCLCommunicator;
  using req_type = cudaStream_t;
  static const req_type null_req;

  /** Return a fresh request. */
  static req_type get_request() {
    cudaStream_t s;
    cudaStreamCreate(&s);
    return s;
  }

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    NonblockingAllreduce(sendbuf, recvbuf, count, op, comm,
                         default_stream, algo);
    comm.synchronize();
  }

  template <typename T>
  static void Allreduce(T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    Allreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingAllreduce(const T* sendbuf, T* recvbuf, size_t count,
                                   ReductionOperator op, comm_type& comm,
                                   req_type& req, algo_type) {
    ncclDataType_t nccl_type = internal::nccl::TypeMap<T>();
    ncclRedOp_t nccl_redop = internal::nccl::ReductionOperator2ncclRedOp(op);
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    if (req == null_req) {
      req = get_request();
    }
    comm.Allreduce((const void*) sendbuf, (void*) recvbuf, count,
                   nccl_type, nccl_redop, req);
  }

  template <typename T>
  static void NonblockingAllreduce(T* recvbuf, size_t count,
                                   ReductionOperator op, comm_type& comm,
                                   req_type& req, algo_type algo) {
    NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                         req, algo);
  }

  template <typename T>
  static void Reduce(const T* sendbuf, T* recvbuf, size_t count,
                     ReductionOperator op, int root, comm_type& comm,
                     algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    NonblockingReduce(sendbuf, recvbuf, count, op, root, comm,
                      default_stream, algo);
    comm.synchronize();
  }

  template <typename T>
  static void Reduce(T* recvbuf, size_t count,
                     ReductionOperator op, int root, comm_type& comm,
                     algo_type algo) {
    Reduce(internal::IN_PLACE<T>(), recvbuf, count, op, root, comm, algo);
  }

  template <typename T>
  static void NonblockingReduce(const T* sendbuf, T* recvbuf, size_t count,
                                ReductionOperator op, int root, comm_type& comm,
                                req_type& req, algo_type) {
    ncclDataType_t nccl_type = internal::nccl::TypeMap<T>();
    ncclRedOp_t nccl_redop = internal::nccl::ReductionOperator2ncclRedOp(op);
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    if (req == null_req) {
      req = get_request();
    }
    comm.Reduce((const void*) sendbuf, (void*) recvbuf, count,
                nccl_type, nccl_redop, root, req);
  }

  template <typename T>
  static void NonblockingReduce(T* recvbuf, size_t count, ReductionOperator op,
                                int root, comm_type& comm, req_type& req,
                                algo_type algo) {
    NonblockingReduce(internal::IN_PLACE<T>(), recvbuf, count, op, root, comm,
                       req, algo);
  }

  template <typename T>
  static void Allgather(const T* sendbuf, T* recvbuf, size_t count,
                        comm_type& comm, algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    NonblockingAllgather(sendbuf, recvbuf, count, comm, default_stream, algo);
    comm.synchronize();
  }

  template <typename T>
  static void Allgather(T* recvbuf, size_t count,
                        comm_type& comm, algo_type algo) {
    Allgather(internal::IN_PLACE<T>(), recvbuf, count, comm, algo);
  }

  template <typename T>
  static void NonblockingAllgather(const T* sendbuf, T* recvbuf, size_t count,
                                   comm_type& comm, req_type& req, algo_type) {
    ncclDataType_t nccl_type = internal::nccl::TypeMap<T>();
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    if (req == null_req) {
      req = get_request();
    }
    comm.Allgather((const void*) sendbuf, (void*) recvbuf, count,
                   nccl_type, req);
  }

  template <typename T>
  static void NonblockingAllgather(T* recvbuf, size_t count, comm_type& comm,
                                   req_type& req, algo_type algo) {
    NonblockingAllgather(internal::IN_PLACE<T>(),  recvbuf, count, comm, req, algo);
  }

  template <typename T>
  static void Reduce_scatter(const T* sendbuf, T* recvbuf, size_t *recv_count,
                             ReductionOperator op, comm_type& comm,
                             algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    NonblockingReduce_scatter(sendbuf, recvbuf, recv_count, op, comm, default_stream, algo);
    comm.synchronize();
  }

  template <typename T>
  static void Reduce_scatter(T* recvbuf, size_t *recv_count,
                             ReductionOperator op, comm_type& comm,
                             algo_type algo) {
    Reduce_scatter(internal::IN_PLACE<T>(), recvbuf, recv_count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingReduce_scatter(const T* sendbuf, T* recvbuf,
                                        size_t *recv_count, ReductionOperator op,
                                        comm_type& comm, req_type& req,
                                        algo_type) {
    // Checking recv_count. This is to fix syntactic mismatch between NCCL and
    // MPI Reduce_scatter.
    size_t r_count = (size_t) recv_count[0];
    bool check = true;
    for (int i = 1; i < comm.size(); i++) {
      if (recv_count[i] != recv_count[0]) {
        check = false;
        break;
      }
    }
    if (!check) {
      if (comm.rank() == 0) {
        std::cerr << "For NCCL_Reduce_scatter recv_count must be equal for all ranks\n";
        std::abort();
      }
    }

    ncclDataType_t nccl_type = internal::nccl::TypeMap<T>();
    ncclRedOp_t nccl_redop = internal::nccl::ReductionOperator2ncclRedOp(op);
    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    if (req == null_req) {
      req = get_request();
    }
    comm.Reduce_scatter((const void*) sendbuf, (void*) recvbuf, r_count,
                        nccl_type, nccl_redop, req);
  }

  template <typename T>
  static void NonblockingReduce_scatter(T* recvbuf, size_t *recv_count,
                                        ReductionOperator op, comm_type& comm,
                                        req_type& req, algo_type algo) {
    NonblockingReduce_scatter(internal::IN_PLACE<T>(), recvbuf, recv_count, op, comm, req, algo);
  }

  template <typename T>
  static void Bcast(T* buf, size_t count, int root, comm_type& comm, 
                    algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    NonblockingBcast(buf, count, root, comm, default_stream, algo);
    comm.synchronize();
  }

  template <typename T>
  static void NonblockingBcast(T* buf, size_t count, int root,
                               comm_type& comm, req_type& req, algo_type) {
    ncclDataType_t nccl_type = internal::nccl::TypeMap<T>();
    if (req == null_req) {
      req = get_request();
    }
    comm.Bcast((void*) buf, count, nccl_type, root, req);
  }
};

template <>
inline bool Test<NCCLBackend>(typename NCCLBackend::req_type& req) {
  if (req == NCCLBackend::null_req) {
    return true;
  }
  return cudaStreamQuery(req) == cudaSuccess;
}

template <>
inline void Wait<NCCLBackend>(typename NCCLBackend::req_type& req) {
  if (req == NCCLBackend::null_req) {
    return;
  }
  cudaStreamSynchronize(req);
}

}  // namespace Al
