/** Benchmark different event implementations. */

#include <iostream>

#include "Al.hpp"
#include "benchmark_utils.hpp"
#include "wait.hpp"

#if defined AL_HAS_ROCM
#include <hip/hip_runtime.h>
#elif defined AL_HAS_CUDA
#include <cuda.h>
#endif

class Event {
public:
  virtual void record(AlGpuStream_t stream) = 0;
  virtual bool query() = 0;
};

class CudaEvent : public Event {
public:
  CudaEvent() {
    AL_CHECK_CUDA(
      AlGpuEventCreateWithFlags(&event, AlGpuNoTimingEventFlags));
  }
  ~CudaEvent() {
    AL_IGNORE_NODISCARD(AlGpuEventDestroy(event));
  }
  void record(AlGpuStream_t stream) override {
    AL_CHECK_CUDA(AlGpuEventRecord(event, stream));
  }
  bool query() override {
    return AlGpuEventQuery(event) == AlGpuSuccess;
  }
private:
  AlGpuEvent_t event;
};

class CustomEvent : public Event {
public:
  CustomEvent() {
    AL_CHECK_CUDA(AlGpuMallocHost(&event, sizeof(int32_t)));
    __atomic_store_n(event, 1, __ATOMIC_SEQ_CST);
#if defined AL_HAS_ROCM
    AL_CHECK_CUDA(hipHostGetDevicePointer(&dev_ptr, event, 0));
#elif defined AL_HAS_CUDA
    AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(&dev_ptr, event, 0));
#endif
  }
  ~CustomEvent() {
    AL_IGNORE_NODISCARD(AlGpuFreeHost(event));
  }
  void record(AlGpuStream_t stream) override {
    __atomic_store_n(event, 0, __ATOMIC_SEQ_CST);
#if defined AL_HAS_ROCM
    AL_CHECK_CUDA(
      hipStreamWriteValue32(stream, dev_ptr, 1, 0));
#elif defined AL_HAS_CUDA
    AL_CHECK_CUDA_DRV(
      cuStreamWriteValue32(stream, dev_ptr, 1, CU_STREAM_WRITE_VALUE_DEFAULT));
#endif
  }
  bool query() override {
    return __atomic_load_n(event, __ATOMIC_SEQ_CST);
  }
private:
  int32_t* event __attribute__((aligned(64)));
#if defined AL_HAS_ROCM
  hipDeviceptr_t dev_ptr;
#elif defined AL_HAS_CUDA
  CUdeviceptr dev_ptr;
#endif
};

void do_benchmark(AlGpuStream_t stream, Event& event) {
  const double wait_time = 0.0001;
  std::vector<double> times, launch_times;
  for (int i = 0; i < 100000; ++i) {
    double launch_start = Al::get_time();
    gpu_wait(wait_time, stream);
    event.record(stream);
    double start = Al::get_time();
    while (!event.query()) {}
    double end = Al::get_time();
    launch_times.push_back(start - launch_start);
    times.push_back(end - start);
    AL_CHECK_CUDA(AlGpuStreamSynchronize(stream));
  }
  std::cout << "Launch: " << SummaryStats(launch_times) << std::endl;
  std::cout << "Query: " << SummaryStats(times) << std::endl;
}

int main(int, char**) {
  AL_CHECK_CUDA(AlGpuSetDevice(0));
  AlGpuStream_t stream;
  AL_CHECK_CUDA(AlGpuStreamCreate(&stream));
  {
    CudaEvent cuda_event;
    CustomEvent custom_event;
    std::cout << "Custom event:" << std::endl;
    do_benchmark(stream, custom_event);
    std::cout << "CUDA Event:" << std::endl;
    do_benchmark(stream, cuda_event);
  }
  AL_CHECK_CUDA(AlGpuStreamSynchronize(stream));
  AL_CHECK_CUDA(AlGpuStreamDestroy(stream));
  return 0;
}
