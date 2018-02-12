#pragma once

namespace allreduces {


class MPIBackend {
 public:
  using algo_type = AllreduceAlgorithm;
  using comm_type = MPICommunicator;
  using req_type = int;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        algo_type algo) {
    if (algo == AllreduceAlgorithm::automatic) {
      // TODO: Better algorithm selection/performance model.
      // TODO: Make tuneable.
      algo = AllreduceAlgorithm::mpi_passthrough;
      if (count <= 1<<9) {
        algo = AllreduceAlgorithm::mpi_recursive_doubling;
      } else {
        algo = AllreduceAlgorithm::mpi_rabenseifner;
      }
    }
    switch (algo) {
      case AllreduceAlgorithm::mpi_passthrough:
        internal::mpi::passthrough_allreduce(sendbuf, recvbuf, count, op, comm);
        break;
      case AllreduceAlgorithm::mpi_recursive_doubling:
        internal::mpi::recursive_doubling_allreduce(
            sendbuf, recvbuf, count, op, comm);
        break;
      case AllreduceAlgorithm::mpi_ring:
        internal::mpi::ring_allreduce(sendbuf, recvbuf, count, op, comm);
        break;
      case AllreduceAlgorithm::mpi_rabenseifner:
        internal::mpi::rabenseifner_allreduce(sendbuf, recvbuf, count, op, comm);
        break;
      case AllreduceAlgorithm::mpi_pe_ring:
        internal::mpi::pe_ring_allreduce(sendbuf, recvbuf, count, op, comm);
        break;
      default:
        throw_allreduce_exception("Invalid algorithm for Allreduce");
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
      req_type& req,
      algo_type algo) {
    if (algo == AllreduceAlgorithm::automatic) {
      // TODO: Better algorithm selection/performance model.
      // TODO: Make tuneable.
      if (count <= 1<<9) {
        algo = AllreduceAlgorithm::mpi_recursive_doubling;
      } else {
        algo = AllreduceAlgorithm::mpi_rabenseifner;
      }
    }
    switch (algo) {
      case AllreduceAlgorithm::mpi_passthrough:
        internal::mpi::nb_passthrough_allreduce(sendbuf, recvbuf, count, op, comm,
                                                req);
        break;
      case AllreduceAlgorithm::mpi_recursive_doubling:
        internal::mpi::nb_recursive_doubling_allreduce(
            sendbuf, recvbuf, count, op, comm, req);
        break;
      case AllreduceAlgorithm::mpi_ring:
        internal::mpi::nb_ring_allreduce(sendbuf, recvbuf, count, op, comm, req);
        break;
      case AllreduceAlgorithm::mpi_rabenseifner:
        internal::mpi::nb_rabenseifner_allreduce(sendbuf, recvbuf, count, op, comm,
                                                 req);
        break;
        /*case AllreduceAlgorithm::mpi_pe_ring:
          internal::mpi::nb_pe_ring_allreduce(sendbuf, recvbuf, count, op, comm, req);
          break;*/
      default:
        throw_allreduce_exception("Invalid algorithm for NonblockingAllreduce");
    }
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
inline bool Test<MPIBackend>(typename MPIBackend::req_type& req) {
  internal::ProgressEngine* pe = internal::get_progress_engine();
  return pe->is_complete(req);
}

template <>
inline void Wait<MPIBackend>(typename MPIBackend::req_type& req) {
  internal::ProgressEngine* pe = internal::get_progress_engine();
  pe->wait_for_completion(req);
}

}  // namespace allreduces


