#pragma once

#include "cuda.hpp"
#include "cudacommunicator.hpp"
#include "mpi_cuda/allreduce_ring.hpp"
#include "mpi_cuda/util.hpp"
#include "mpi_impl.hpp"
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
  comm.get_ring().allreduce<T>(recvbuf, count, op, stream, false);
}

template <typename T> inline
void bi_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, MPICUDACommunicator& comm,
                       cudaStream_t stream) {
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * count,
                                    cudaMemcpyDefault, stream));
  }
  comm.get_ring().allreduce<T>(recvbuf, count, op, stream, true);
}

struct host_transfer_allreduce_data {
  host_transfer_allreduce_data(void* data_, size_t count_,
                               ReductionOperator op_, MPICommunicator& comm_,
                               int tag_) :
    data(data_), count(count_), op(op_), comm(comm_), tag(tag_) {}
  void* data;
  size_t count;
  ReductionOperator op;
  MPICommunicator& comm;
  int tag;
};

template <typename T> inline
void host_transfer_allreduce_callback(cudaStream_t, cudaError_t,
                                      void* data_) {
  host_transfer_allreduce_data* data =
    static_cast<host_transfer_allreduce_data*>(data_);
  // Use a tag to prevent interference.
  MPIBackend::Allreduce<T>(IN_PLACE<T>(), (T*) data->data, data->count,
                           data->op, data->comm, AllreduceAlgorithm::automatic,
                           data->tag);
}

template <typename T> inline
void host_transfer_allreduce_free_mem(cudaStream_t, cudaError_t, void* data_) {
  T* host_mem = static_cast<T*>(data_);
  release_pinned_memory(host_mem);
}

template <typename T> inline
void host_transfer_allreduce(const T* sendbuf, T* recvbuf, T* host_mem,
                             size_t count, ReductionOperator op,
                             MPICUDACommunicator& comm, cudaStream_t stream,
                             int tag) {
  // If not in-place, transfer the sendbuf.
  if (sendbuf != recvbuf) {
    COLL_CHECK_CUDA(cudaMemcpyAsync(host_mem, sendbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
  } else {
    COLL_CHECK_CUDA(cudaMemcpyAsync(host_mem, recvbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
  }
  // Launch callback to run allreduce.
  host_transfer_allreduce_data* data = new host_transfer_allreduce_data(
    host_mem, count, op, comm, tag);
  COLL_CHECK_CUDA(cudaStreamAddCallback(stream,
                                        host_transfer_allreduce_callback<T>,
                                        (void*) data, 0));
  // This will transfer the final data back to device.
  COLL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
}

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
