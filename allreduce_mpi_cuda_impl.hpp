#pragma once

namespace allreduces {

enum class MPICUDAAllreduceAlgorithm {
  automatic,
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

namespace internal {
namespace mpi_cuda {

template <typename T>
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, MPICommunicator& comm) {
}

template <typename T>
void bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICommunicator& comm) {
}

template <typename T>
void nb_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICommunicator& comm,
                       AllreduceRequest& req) {
}

template <typename T>
void nb_bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                          ReductionOperator op, MPICommunicator& comm,
                          AllreduceRequest& req) {
}

} // namespace mpi_cuda
} // namespace internal

class MPICUDABackend {
 public:
  using algo_type = MPICUDAAllreduceAlgorithm;
  using comm_type = MPICommunicator;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    switch (algo) {
      case MPICUDAAllreduceAlgorithm::ring:
        internal::mpi_cuda::ring_allreduce(sendbuf, recvbuf, count,
                                           op, comm);
        break;
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
    Allreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm, algo);
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
