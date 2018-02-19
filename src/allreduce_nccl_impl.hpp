#pragma once

#include "allreduce.hpp"
#include "nccl.h"
#include "common.h"

namespace allreduces {

enum class NCCLAllreduceAlgorithm {
  automatic
};

inline std::string allreduce_name(NCCLAllreduceAlgorithm algo) {
  switch (algo) {
  case NCCLAllreduceAlgorithm::automatic:
    return "automatic";
  default:
    return "unknown";
  }
}

/// We assume NCCL version 2.0 or higher for allreduce to work
class NCCLCommunicator : public MPICommunicator {
 public:
  /** Default constructor; use MPI_COMM_WORLD. */
  NCCLCommunicator() : NCCLCommunicator(MPI_COMM_WORLD) {}
  /** Use a particular MPI communicator. */
  NCCLCommunicator(MPI_Comm comm_) : MPICommunicator(comm_) {
    MPI_Comm_dup(comm_, &mpi_comm);

    /// Set up GPU-related informatiton
    gpu_setup();

    /// NCCL set up here
    nccl_setup();
  }

  ~NCCLCommunicator() override {
    nccl_destroy();
  }

  Communicator* copy() const override { return new NCCLCommunicator(mpi_comm); }

  void synchronize();

  cudaStream_t get_default_stream();

  /// It is assumed that both sendbuf and recvbuf are in device memory
  /// for NCCL sendbuf and recvbuf can be identical; in-place operation will be performed
  void Allreduce(void* sendbuf, void* recvbuf, size_t count, ncclDataType_t nccl_type,
               ncclRedOp_t nccl_redop); 

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
  /// Number of visible GPUs on this compute node
  int m_num_visible_gpus;

  /** List of NCCL 2 related variables. */
  /// NOTE: It is assumed that ONLY ONE GPU is allocated to one MPI rank
  std::vector<ncclComm_t> m_nccl_comm;
};

class NCCLBackend {
 public:
  using algo_type = NCCLAllreduceAlgorithm;
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
      algo_type) {
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
                   nccl_type, nccl_redop);
                   //nccl_type, nccl_redop, req);
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

