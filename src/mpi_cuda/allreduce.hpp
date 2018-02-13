#pragma once

#include "mpi_cuda/allreduce_ring.hpp"
#include "mpi_cuda/util.hpp"
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
    COLL_CHECK_CUDA(cudaMemcpy(recvbuf, sendbuf, sizeof(T) * count,
                               cudaMemcpyDefault));
  }
  comm.get_ring().allreduce<T>({recvbuf}, count, nullptr, false);
}

template <typename T> inline
void bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm) {
  assert(op == ReductionOperator::sum);  
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpy(recvbuf, sendbuf, sizeof(T) * count,
                               cudaMemcpyDefault));
  }
  comm.get_ring().allreduce<T>({recvbuf}, count, nullptr, true);
}

template <typename T> inline
void nb_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm,
                       cudaStream_t &stream) {
  assert(op == ReductionOperator::sum);  
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  std::vector<cudaStream_t> streams = {stream};
  comm.get_ring().allreduce<T>({recvbuf}, count, &streams, false);
}

template <typename T> inline
void nb_bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                          ReductionOperator op, MPICUDACommunicator& comm,
                          cudaStream_t &stream) {
  assert(op == ReductionOperator::sum);  
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  std::vector<cudaStream_t> streams = {stream};  
  comm.get_ring().allreduce<T>({recvbuf}, count, &streams, true);
}


} // namespace mpi_cuda
} // namespace internal
} // namespace allreduces

