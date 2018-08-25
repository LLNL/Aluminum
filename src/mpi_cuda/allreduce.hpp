#pragma once

#include "cuda.hpp"
#include "progress.hpp"
#include "mpi_cuda/communicator.hpp"
#include "mpi_cuda/allreduce_ring.hpp"
#include "mpi_cuda/util.hpp"
#include "mpi_impl.hpp"
#include <cassert>

namespace Al {
namespace internal {
namespace mpi_cuda {

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

/** Progress engine state for the host-transfer allreduce. */
template <typename T>
class HostTransferState : public AlState {
 public:
  HostTransferState(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, MPICUDACommunicator& comm,
                    cudaStream_t stream, AlRequest req_) : AlState(req_) {
    host_mem = get_pinned_memory<T>(count);
    if (count <= 1<<9) {
      host_ar = new mpi::MPIRecursiveDoublingAlState<T>(
        IN_PLACE<T>(), host_mem, count, op, comm, get_free_request());
    } else {
      host_ar = new mpi::MPIRabenseifnerAlState<T>(
        IN_PLACE<T>(), host_mem, count, op, comm, get_free_request());
    }
    sync_event = cuda::get_cuda_event();
    sync_event2 = cuda::get_cuda_event();
    // The device will sync on this memory location.
    sync = get_pinned_memory<int32_t>(1);
    *sync = 0;
    AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(&sync_dev_ptr, sync, 0));
    // Transfer data from device to host and use an event to determine when it
    // completes. Handle in-place vs non-in-place.
    if (sendbuf != recvbuf) {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem, sendbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
    } else {
      AL_CHECK_CUDA(cudaMemcpyAsync(host_mem, recvbuf, sizeof(T)*count,
                                    cudaMemcpyDeviceToHost, stream));
    }
    AL_CHECK_CUDA(cudaEventRecord(sync_event, stream));
    // Have the device wait on the host.
    AL_CHECK_CUDA_DRV(cuStreamWaitValue32(stream, sync_dev_ptr, 1,
                                          CU_STREAM_WAIT_VALUE_EQ));
    // Transfer completed buffer back to device.
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, host_mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
    AL_CHECK_CUDA(cudaEventRecord(sync_event2, stream));
  }

  ~HostTransferState() {
    
  }

  bool step() override {
    if (!ar_started) {
      // Wait for memory to get to the host.
      cudaError_t r = cudaEventQuery(sync_event);
      if (r == cudaSuccess) {
        host_ar->setup();
        ar_started = true;
      } else if (r == cudaErrorNotReady) {
        return false;
      } else {
        throw_al_exception("cudaEventQuery error");
      }
    }
    if (!ar_done) {
      // Wait for the allreduce to complete.
      if (host_ar->step()) {
        ar_done = true;
        delete host_ar;  // TODO: Maybe move this.
        // Mark the sync as done to wake up the device.
        *sync = 1;
      } else {
        return false;
      }
    }
    // Wait for the memcpy back to device to complete so we can clean up.
    cudaError_t r = cudaEventQuery(sync_event2);
    if (r == cudaSuccess) {
      release_pinned_memory(host_mem);  // TODO: Maybe move this.
      release_pinned_memory(sync);
      cuda::release_cuda_event(sync_event);
      cuda::release_cuda_event(sync_event2);
      return true;
    } else if (r != cudaErrorNotReady) {
      throw_al_exception("cudaEventQuery error");
    }
    return false;
  }
  bool needs_completion() const override { return false; }
 private:
  cudaEvent_t sync_event;
  cudaEvent_t sync_event2;
  bool ar_started = false;
  bool ar_done = false;
  mpi::MPIAlState<T>* host_ar;
  T* host_mem;
  int32_t* sync;
  CUdeviceptr sync_dev_ptr;
};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
