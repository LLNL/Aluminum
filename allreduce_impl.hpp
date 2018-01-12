#pragma once

namespace allreduces {

template <typename T>
void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
               ReductionOperator op, Communicator& comm,
               AllreduceAlgorithm algo) {
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
void Allreduce(T* recvbuf, size_t count,
               ReductionOperator op, Communicator& comm,
               AllreduceAlgorithm algo) {
  Allreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm, algo);
}

template <typename T>
void NonblockingAllreduce(
  const T* sendbuf, T* recvbuf, size_t count,
  ReductionOperator op, Communicator& comm,
  AllreduceRequest& req,
  AllreduceAlgorithm algo) {
  if (algo == AllreduceAlgorithm::automatic) {
    // TODO: Algorithm selection/performance model.
    algo = AllreduceAlgorithm::mpi_recursive_doubling;
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
void NonblockingAllreduce(
  T* recvbuf, size_t count,
  ReductionOperator op, Communicator& comm,
  AllreduceRequest& req,
  AllreduceAlgorithm algo) {
  NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                       req, algo);
}

}  // namespace allreduces
