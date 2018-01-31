#pragma once

namespace allreduces {

template <typename T>
void NCCLAllreduce(const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op, Communicator& comm){

  allreduces::NCCLCommunicator& xcomm = dynamic_cast<NCCLCommunicator&>(comm);

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

  xcomm.Allreduce((void*) sendbuf, (void*) recvbuf, count, nccl_type, nccl_redop);


}

template <typename T>
void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
               ReductionOperator op, Communicator& comm,
               AllreduceAlgorithm algo) {


  /// Regular MPI-based allreduce
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
void NonblockingAllreduce(
  T* recvbuf, size_t count,
  ReductionOperator op, Communicator& comm,
  AllreduceRequest& req,
  AllreduceAlgorithm algo) {
  NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                       req, algo);
}
}  // namespace allreduces


