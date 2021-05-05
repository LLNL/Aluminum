/** Benchmark different wait implementations. */

#include <iostream>
#include <cuda.h>
#include "Al.hpp"
#include "aluminum/cuda/helper_kernels.hpp"
#include "benchmark_utils.hpp"
#include "wait.hpp"

class Wait {
public:
  Wait() {
    cudaMallocHost(&wait_sync, sizeof(int32_t));
    __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);
  }
  ~Wait() {
    cudaFreeHost(wait_sync);
  }
  virtual void wait(cudaStream_t stream) = 0;
  virtual void signal() {
    __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST);
  }

  int32_t* wait_sync __attribute__((aligned(64)));
};

class StreamOpWait : public Wait {
public:
  StreamOpWait() : Wait() {
    cuMemHostGetDevicePointer(&dev_ptr, wait_sync, 0);
  }
  ~StreamOpWait() {}
  void wait(cudaStream_t stream) override {
    Al::internal::cuda::launch_wait_kernel(stream, 1, dev_ptr);
  }
  CUdeviceptr dev_ptr;
};

class KernelWait : public Wait {
public:
  KernelWait() : Wait() {
    cudaHostGetDevicePointer(&dev_ptr, wait_sync, 0);
  }
  ~KernelWait() {}
  void wait(cudaStream_t stream) override {
    Al::internal::cuda::launch_wait_kernel(stream, 1, dev_ptr);
  }
  int32_t* dev_ptr __attribute__((aligned(64)));
};

void do_benchmark(cudaStream_t stream, Wait& wait) {
  cudaEvent_t e;
  cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
  std::vector<double> times, launch_times;
  for (int i = 0; i < 100000; ++i) {
    double launch_start = Al::get_time();
    wait.wait(stream);
    double launch_end = Al::get_time();
    cudaEventRecord(e, stream);
    double start = Al::get_time();
    wait.signal();
    while (cudaEventQuery(e) == cudaErrorNotReady) {}
    double end = Al::get_time();
    launch_times.push_back(launch_end - launch_start);
    times.push_back(end - start);
    cudaStreamSynchronize(stream);
  }
  std::cout << "Launch: " << SummaryStats(launch_times) << std::endl;
  std::cout << "Signal: " << SummaryStats(times) << std::endl;
  cudaEventDestroy(e);
}

int main(int, char**) {
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  {
    StreamOpWait stream_op_wait;
    KernelWait kernel_wait;
    std::cout << "StreamOp wait:" << std::endl;
    do_benchmark(stream, stream_op_wait);
    std::cout << "Kernel wait:" << std::endl;
    do_benchmark(stream, kernel_wait);
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return 0;
}
