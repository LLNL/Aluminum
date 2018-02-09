#pragma once

#include "mpi_cuda/allreduce_ring.hpp"
#include <cassert>

namespace allreduces {
namespace internal {
namespace mpi_cuda {

class MPICUDACommunicator: public MPICommunicator {
 public:
  MPICUDACommunicator(MPI_Comm c=MPI_COMM_WORLD):
      MPICommunicator(c), m_ring(get_comm()) {}
  
  RingMPICUDA &get_ring() {
    return m_ring;
  }
  
 protected:
  RingMPICUDA m_ring;
};

template <typename T> inline
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, MPICUDACommunicator& comm) {
  assert(op == ReductionOperator::sum);
  if (sendbuf != recvbuf) {
    cudaMemcpy(recvbuf, sendbuf, sizeof(T) * count, cudaMemcpyDefault);
  }
  comm.get_ring().allreduce<T>({recvbuf}, count, nullptr, false);
}

template <typename T> inline
void bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm) {
  assert(op == ReductionOperator::sum);  
  if (sendbuf != recvbuf) {
    cudaMemcpy(recvbuf, sendbuf, sizeof(T) * count, cudaMemcpyDefault);
  }
  comm.get_ring().allreduce<T>({recvbuf}, count, nullptr, true);
}

// TODO: This is actually blocking
template <typename T> inline
void nb_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm,
                       AllreduceRequest&) {
  assert(op == ReductionOperator::sum);  
  if (sendbuf != recvbuf) {
    cudaMemcpy(recvbuf, sendbuf, sizeof(T) * count, cudaMemcpyDefault);
  }
  comm.get_ring().allreduce<T>({recvbuf}, count, nullptr, false);
}

// TODO: This is actually blocking
template <typename T> inline
void nb_bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                          ReductionOperator op, MPICUDACommunicator& comm,
                          AllreduceRequest&) {
  assert(op == ReductionOperator::sum);  
  if (sendbuf != recvbuf) {
    cudaMemcpy(recvbuf, sendbuf, sizeof(T) * count, cudaMemcpyDefault);
  }
  comm.get_ring().allreduce<T>({recvbuf}, count, nullptr, true);
}


} // namespace mpi_cuda
} // namespace internal
} // namespace allreduces

