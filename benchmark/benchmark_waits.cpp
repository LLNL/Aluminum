/** Benchmark different wait implementations. */

#include <iostream>
#include "Al.hpp"
#include "aluminum/cuda/helper_kernels.hpp"
#include "benchmark_utils.hpp"
#include "wait.hpp"

#if defined AL_HAS_ROCM
#include <hip/hip_runtime.h>
#elif defined AL_HAS_CUDA
#include <cuda.h>
#endif

class Wait {
public:
  Wait() {
    AL_IGNORE_NODISCARD(AlGpuMallocHost(&wait_sync, sizeof(int32_t)));
    __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);
  }
  ~Wait() {
    AL_IGNORE_NODISCARD(AlGpuFreeHost(wait_sync));
  }
  virtual void wait(AlGpuStream_t stream) = 0;
  virtual void signal() {
    __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST);
  }

  int32_t* wait_sync __attribute__((aligned(64)));
};

#if defined AL_HAS_ROCM
class StreamOpWait : public Wait {
public:
  StreamOpWait() : Wait() {
    AL_IGNORE_NODISCARD(hipHostGetDevicePointer(&dev_ptr, wait_sync, 0));
  }
  ~StreamOpWait() {}
  void wait(AlGpuStream_t stream) override {
    Al::internal::cuda::launch_wait_kernel(stream, 1, dev_ptr);
  }
  hipDeviceptr_t dev_ptr;
};
#elif defined AL_HAS_CUDA
class StreamOpWait : public Wait {
public:
  StreamOpWait() : Wait() {
    AL_IGNORE_NODISCARD(cuMemHostGetDevicePointer(&dev_ptr, wait_sync, 0));
  }
  ~StreamOpWait() {}
  void wait(AlGpuStream_t stream) override {
    Al::internal::cuda::launch_wait_kernel(stream, 1, dev_ptr);
  }
  CUdeviceptr dev_ptr;
};
#endif

class KernelWait : public Wait {
public:
  KernelWait() : Wait() {
    AL_IGNORE_NODISCARD(AlGpuHostGetDevicePointer(
                          reinterpret_cast<void **>(&dev_ptr), wait_sync, 0));
  }
  ~KernelWait() {}
  void wait(AlGpuStream_t stream) override {
    Al::internal::cuda::launch_wait_kernel(stream, 1, dev_ptr);
  }
  int32_t* dev_ptr __attribute__((aligned(64)));
};

void do_benchmark(AlGpuStream_t stream, Wait& wait) {
  AlGpuEvent_t e;
  AL_IGNORE_NODISCARD(AlGpuEventCreateWithFlags(&e, AlGpuEventDisableTiming));
  std::vector<double> times, launch_times;
  for (int i = 0; i < 100000; ++i) {
    double launch_start = Al::get_time();
    wait.wait(stream);
    double launch_end = Al::get_time();
    AL_IGNORE_NODISCARD(AlGpuEventRecord(e, stream));
    double start = Al::get_time();
    wait.signal();
    while (AlGpuEventQuery(e) == AlGpuErrorNotReady) {}
    double end = Al::get_time();
    launch_times.push_back(launch_end - launch_start);
    times.push_back(end - start);
    AL_IGNORE_NODISCARD(AlGpuStreamSynchronize(stream));
  }
  std::cout << "Launch: " << SummaryStats(launch_times) << std::endl;
  std::cout << "Signal: " << SummaryStats(times) << std::endl;
  AL_IGNORE_NODISCARD(AlGpuEventDestroy(e));
}

int main(int, char**) {
  AL_IGNORE_NODISCARD(AlGpuSetDevice(0));
  AlGpuStream_t stream;
  AL_IGNORE_NODISCARD(AlGpuStreamCreate(&stream));
  {
    StreamOpWait stream_op_wait;
    KernelWait kernel_wait;
    std::cout << "StreamOp wait:" << std::endl;
    do_benchmark(stream, stream_op_wait);
    std::cout << "Kernel wait:" << std::endl;
    do_benchmark(stream, kernel_wait);
  }
  AL_IGNORE_NODISCARD(AlGpuStreamSynchronize(stream));
  AL_IGNORE_NODISCARD(AlGpuStreamDestroy(stream));
  return 0;
}
