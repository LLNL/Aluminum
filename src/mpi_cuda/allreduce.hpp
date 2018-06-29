#pragma once

#include "cuda.hpp"
#include "cudacommunicator.hpp"
#include "mpi_cuda/allreduce_ring.hpp"
#include "mpi_cuda/util.hpp"
#include <cassert>

namespace Al {
namespace internal {
namespace mpi_cuda {

class MPICUDACommunicator: public CUDACommunicator {
 public:
  MPICUDACommunicator() : MPICUDACommunicator(MPI_COMM_WORLD, 0) {}
  MPICUDACommunicator(cudaStream_t stream_) :
    MPICUDACommunicator(MPI_COMM_WORLD, stream_) {}
  MPICUDACommunicator(MPI_Comm comm_, cudaStream_t stream_) :
    CUDACommunicator(comm_, stream_), m_ring(get_comm()) {}
  
  RingMPICUDA &get_ring() {
    return m_ring;
  }
  
 protected:
  RingMPICUDA m_ring;
};

template <typename T> inline
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm,
                       cudaStream_t stream) {
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  std::vector<cudaStream_t> streams = {stream};
  comm.get_ring().allreduce<T>({recvbuf}, count, op, &streams, false);
}

template <typename T> inline
void bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                          ReductionOperator op, MPICUDACommunicator& comm,
                          cudaStream_t stream) {
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  std::vector<cudaStream_t> streams = {stream};  
  comm.get_ring().allreduce<T>({recvbuf}, count, op, &streams, true);
}

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
