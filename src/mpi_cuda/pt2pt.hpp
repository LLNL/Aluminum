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
      }
      // Always return false here so the send is not started until the next
      // pass through the in-progress requests.
      // This ensures that sends always start in the order they were posted.
      return false;
    }
    if (!send_started) {
      MPI_Isend(mem, count, mpi::TypeMap<T>(), dest, pt2pt_tag, comm, &req);
      send_started = true;
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
  bool send_started = false;
};

template <typename T>
class RecvAlState : public AlState {
 public:
  RecvAlState(T* recvbuf, size_t count_, int src_,
              MPICUDACommunicator& comm_, cudaStream_t stream) :
    AlState(nullptr), count(count_), src(src_), comm(comm_.get_comm()) {
    mem = get_pinned_memory<T>(count);
    wait_sync.wait(stream);
    AL_CHECK_CUDA(cudaMemcpyAsync(recvbuf, mem, sizeof(T)*count,
                                  cudaMemcpyHostToDevice, stream));
    sync_event.record(stream);
  }
  ~RecvAlState() override {
    release_pinned_memory(mem);
  }
  void start() override {
    MPI_Irecv(mem, count, mpi::TypeMap<T>(), src, pt2pt_tag, comm, &req);
  }
  bool step() override {
    if (!recv_done) {
      int flag;
      MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        recv_done = true;
        // Signal the device that the memcpy can start.
        wait_sync.signal();
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
  cuda::GPUWait wait_sync;
  bool recv_done = false;
};

template <typename T>
class SendRecvAlState : public AlState {
 public:
  SendRecvAlState(const T* sendbuf, size_t send_count, int dest,
                  T* recvbuf, size_t recv_count, int src,
                  MPICUDACommunicator& comm, cudaStream_t stream) :
    AlState(nullptr),
    send_state(sendbuf, send_count, dest, comm, stream),
    recv_state(recvbuf, recv_count, src, comm, stream) {}
  void start() override {
    send_state.start();
    recv_state.start();
  }
  bool step() override {
    if (!send_done) {
      send_done = send_state.step();
    }
    if (!recv_done) {
      recv_done = recv_state.step();
    }
    return send_done && recv_done;
  }
  bool needs_completion() const override { return false; }
 private:
  SendAlState<T> send_state;
  RecvAlState<T> recv_state;
  bool send_done = false;
  bool recv_done = false;
};

}  // namespace mpi_cuda
}  // namespace internal
}  // namespace Al
