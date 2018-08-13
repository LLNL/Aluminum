#pragma once

#include "cuda.hpp"
#include "mpi_cuda/allreduce.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

constexpr int pt2pt_tag = 2;

/** GPU point-to-point send operation. */
template <typename T>
class SendAlState : public AlState {
 public:
  SendAlState(const T* sendbuf, size_t count_, int dest_,
              MPICUDACommunicator& comm_, cudaStream_t stream) :
    AlState(nullptr), count(count_), dest(dest_), comm(comm_.get_comm()) {
    mem = get_pinned_memory<T>(count);
    AL_CHECK_CUDA(cudaMemcpyAsync(mem, sendbuf, sizeof(T)*count,
                                  cudaMemcpyDeviceToHost, stream));
    sync_event.record(stream);
  }
  ~SendAlState() override {
    release_pinned_memory(mem);
  }
  bool step() override {
    if (!mem_transfer_done) {
      if (sync_event.query()) {
        mem_transfer_done = true;
        MPI_Isend(mem, count, mpi::TypeMap<T>(), dest, pt2pt_tag, comm, &req);
      } else {
        return false;
      }
    }
    int flag;
    MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
    return flag;
  }
  bool needs_completion() const override { return false; }
 private:
  T* mem;
  size_t count;
  int dest;
  MPI_Comm comm;
  MPI_Request req = MPI_REQUEST_NULL;
  cuda::FastEvent sync_event;
  bool mem_transfer_done = false;
};

template <typename T>
class RecvAlState : public AlState {
 public:
  RecvAlState(T* recvbuf, size_t count_, int src_,
              MPICUDACommunicator& comm_, cudaStream_t stream) :
    AlState(nullptr), count(count_), src(src_), comm(comm_.get_comm()) {
    mem = get_pinned_memory<T>(count);
    wait_sync = get_pinned_memory<int32_t>(1);
    __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);
    AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(
                        &wait_sync_dev_ptr, wait_sync, 0));
    AL_CHECK_CUDA_DRV(cuStreamWaitValue32(
                        stream, wait_sync_dev_ptr, 1, CU_STREAM_WAIT_VALUE_EQ));
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
    sync_event.record(stream);
  }
  ~RecvAlState() override {
    release_pinned_memory(mem);
    release_pinned_memory(wait_sync);
  }
  bool step() override {
    if (!recv_started) {
      MPI_Irecv(mem, count, mpi::TypeMap<T>(), src, pt2pt_tag, comm, &req);
      recv_started = true;
    }
    if (!recv_done) {
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        recv_done = true;
        // Signal the device that the memcpy can start.
        __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST);
      }
    }
    // Wait until the memcpy has completed so everything can be safely freed.
    return sync_event.query();
  }
  bool needs_completion() const override { return false; }
 private:
  T* mem;
  size_t count;
  int src;
  MPI_Comm comm;
  MPI_Request req = MPI_REQUEST_NULL;
  cuda::FastEvent sync_event;
  int32_t* wait_sync __attribute__((aligned(64)));
  CUdeviceptr wait_sync_dev_ptr;
  bool recv_started = false;
  bool recv_done = false;
};

}  // namespace mpi_cuda
}  // namespace internal
}  // namespace Al
