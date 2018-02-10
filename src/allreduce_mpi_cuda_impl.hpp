#pragma once

#include "mpi_cuda/allreduce.hpp"

namespace allreduces {

enum class MPICUDAAllreduceAlgorithm {
  automatic,
  // TODO: unidirectional ring does not work; will fix
  ring, 
  bi_ring
};

inline std::string allreduce_name(MPICUDAAllreduceAlgorithm algo) {
  switch (algo) {
  case MPICUDAAllreduceAlgorithm::automatic:
    return "automatic";
  case MPICUDAAllreduceAlgorithm::ring:
    return "ring";
  case MPICUDAAllreduceAlgorithm::bi_ring:
    return "bi-ring";
  default:
    return "unknown";
  }
}

class MPICUDABackend {
 public:
  using algo_type = MPICUDAAllreduceAlgorithm;
  using comm_type = internal::mpi_cuda::MPICUDACommunicator;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    switch (algo) {
      case MPICUDAAllreduceAlgorithm::ring:
        internal::mpi_cuda::ring_allreduce(sendbuf, recvbuf, count,
                                           op, comm);
        break;
      case MPICUDAAllreduceAlgorithm::automatic:
      case MPICUDAAllreduceAlgorithm::bi_ring:
        internal::mpi_cuda::bi_ring_allreduce(sendbuf, recvbuf, count,
                                              op, comm);
        break;
      default:
        throw_allreduce_exception("Invalid algorithm");
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
      AllreduceRequest& req,
      algo_type algo) {
    switch (algo) {
      case MPICUDAAllreduceAlgorithm::ring:
        internal::mpi_cuda::nb_ring_allreduce(sendbuf, recvbuf, count,
                                              op, comm, req);
        break;
      case MPICUDAAllreduceAlgorithm::bi_ring:
        internal::mpi_cuda::nb_bi_ring_allreduce(sendbuf, recvbuf, count,
                                                 op, comm, req);
        break;
      default:
        throw_allreduce_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingAllreduce(
      T* recvbuf, size_t count,
      ReductionOperator op, comm_type& comm,
      AllreduceRequest& req,
      algo_type algo) {
    NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                         req, algo);
  }

};

} // namespace allreduces
