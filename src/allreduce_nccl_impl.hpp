#pragma once

#include "allreduce.hpp"
#include "nccl.h"
#include "common.h"

namespace allreduces {

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

/// We assume NCCL version 2.0 or higher for allreduce to work
class NCCLCommunicator : public MPICommunicator {
 public:
  NCCLCommunicator(MPI_Comm comm_ = MPI_COMM_WORLD,
                   std::vector<int> gpus = std::vector<int>());

  ~NCCLCommunicator() override;

  Communicator* copy() const override { return new NCCLCommunicator(mpi_comm); }

  void synchronize();

  cudaStream_t get_default_stream();

  /// It is assumed that both sendbuf and recvbuf are in device memory
  /// for NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  void Allreduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
               ncclRedOp_t nccl_redop, cudaStream_t default_stream); 

  void Bcast(void* sendbuf, size_t count, ncclDataType_t nccl_type, int root, 
               cudaStream_t default_stream); 

  void Reduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
               ncclRedOp_t nccl_redop, int root, cudaStream_t default_stream); 

  /**
  * It is assumed that both sendbuf and recvbuf are in device memory
  * For NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  * NCCL-based Allgather assumes that recv_count is equal to num_ranks*send_count, which means
  * that recv_buf should have a size of at least num_ranks*send_count elements
  */
  void Allgather(void* sendbuf, void* recvbuf, size_t count,
                 ncclDataType_t nccl_type, cudaStream_t default_stream);

  /**
  * It is assumed that both sendbuf and recvbuf are in device memory
  * For NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  * NCCL-based Reduce_scatter assumes that send_count is equal to num_ranks*recv_count, which means
  * that send_buf should have a size of at least num_ranks*recv_count elements
  */
  void Reduce_scatter(void* sendbuf, void* recvbuf, size_t recv_count, ncclDataType_t nccl_type,
                     ncclRedOp_t nccl_redop, cudaStream_t default_stream);


protected:

  void gpu_setup();

  void nccl_setup();

  void nccl_destroy();

 private:

  MPI_Comm mpi_comm;

  /** List of GPU related variables. */
  /// List of GPUs to be used
  std::vector<int> m_gpus;
  /// List of CUDA streams
  std::vector<cudaStream_t> m_streams;
  /// Number of GPUs allocated to the current rank
  int m_num_gpus;

  /** List of NCCL 2 related variables. */
  /// NOTE: It is assumed that ONLY ONE GPU is allocated to one MPI rank
  ncclComm_t m_nccl_comm;
};

class NCCLBackend {
 public:
  using algo_type = NCCLCollectiveAlgorithm;
  using comm_type = NCCLCommunicator;
  using req_type = cudaStream_t;

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
  static void NonblockingAllreduce(
      const T* sendbuf, T* recvbuf, size_t count,
      ReductionOperator op,
      comm_type& comm,
      req_type& req,
      algo_type algo) {
    ncclDataType_t nccl_type;
    switch(sizeof(T)) {
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
        throw_allreduce_exception("unsupported NCCL data type");
    }

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
        throw_allreduce_exception("unsupported NCCL reduction operator");
    }

    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    
    comm.Allreduce((void*) sendbuf, (void*) recvbuf, count,
                   nccl_type, nccl_redop, req);
  }

  template <typename T>
  static void NonblockingAllreduce(
      T* recvbuf, size_t count,
      ReductionOperator op, comm_type& comm,
      req_type& req,
      algo_type algo) {
    NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                         req, algo);
  }

  template <typename T>
  static void Bcast(const T* sendbuf, size_t count, int root, comm_type& comm, algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    ncclDataType_t nccl_type;
    switch(sizeof(T)) {
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
        throw_allreduce_exception("unsupported NCCL data type");
    }

    Bcast(sendbuf, count, nccl_type, root, default_stream);
    comm.synchronize();
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
  static void NonblockingReduce(
      const T* sendbuf, T* recvbuf, size_t count,
      ReductionOperator op,
      int root,
      comm_type& comm,
      req_type& req,
      algo_type algo) {

    ncclDataType_t nccl_type;
    switch(sizeof(T)) {
      case 8:
        nccl_type = ncclDouble;
        break;
      case 4:
        nccl_type = ncclFloat;
        break;
      case 2:
        nccl_type = ncclHalf;
        break;
      default: throw_allreduce_exception("unsupported NCCL data type");
    }

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
        throw_allreduce_exception("unsupported NCCL reduction operator");
    }

    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    
    comm.Reduce((void*) sendbuf, (void*) recvbuf, count,
                nccl_type, nccl_redop, root, req);
  }

  template <typename T>
  static void NonblockingReduce(
      T* recvbuf, size_t count,
      ReductionOperator op, 
      int root, 
      comm_type& comm,
      req_type& req,
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
  static void NonblockingAllgather(
      const T* sendbuf, T* recvbuf, size_t count,
      comm_type& comm,
      req_type& req,
      algo_type algo) {

    ncclDataType_t nccl_type;
    switch(sizeof(T)) {
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
        throw_allreduce_exception("unsupported NCCL data type");
    }

    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }

    comm.Allgather((void*) sendbuf, (void*) recvbuf, count,
                   nccl_type, req);
  }

  template <typename T>
  static void NonblockingAllgather(
      T* recvbuf, size_t count,
      comm_type& comm,
      req_type& req,
      algo_type algo) {
  
    NonblockingAllgather(internal::IN_PLACE<T>(),  recvbuf, count, comm, req, algo);

  }

  template <typename T>
  static void Reduce_scatter(const T* sendbuf, T* recvbuf, size_t recv_count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    cudaStream_t default_stream = comm.get_default_stream();
    NonblockingReduce_scatter(sendbuf, recvbuf, recv_count, op, comm, default_stream, algo);
    comm.synchronize();
  }

  template <typename T>
  static void Reduce_scatter(T* recvbuf, size_t recv_count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    Reduce_scatter(internal::IN_PLACE<T>(), recvbuf, recv_count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingReduce_scatter(
      const T* sendbuf, T* recvbuf, size_t recv_count,
      ReductionOperator op,
      comm_type& comm,
      req_type& req,
      algo_type algo) {

    ncclDataType_t nccl_type;
    switch(sizeof(T)) {
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
        throw_allreduce_exception("unsupported NCCL data type");
    }

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
        throw_allreduce_exception("unsupported NCCL reduction operator");
    }

    if (sendbuf == internal::IN_PLACE<T>()) {
      sendbuf = recvbuf;
    }
    
    comm.Reduce_scatter((void*) sendbuf, (void*) recvbuf, recv_count,
                   nccl_type, nccl_redop, req);
  }

  template <typename T>
  static void NonblockingReduce_scatter(
      T* recvbuf, size_t recv_count,
      ReductionOperator op, comm_type& comm,
      req_type& req,
      algo_type algo) {
    NonblockingReduce_scatter(internal::IN_PLACE<T>(), recvbuf, recv_count, op, comm, req, algo);
  }

};


template <>
inline bool Test<NCCLBackend>(typename NCCLBackend::req_type& req) {
  return cudaStreamQuery(req) == cudaSuccess;
}

template <>
inline void Wait<NCCLBackend>(typename NCCLBackend::req_type& req) {
  cudaStreamSynchronize(req);
}

}  // namespace allreduces


