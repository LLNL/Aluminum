/** Benchmark different event implementations. */

#include <iostream>
#include <cuda.h>
#include "Al.hpp"
#include "benchmark_utils.hpp"
#include "wait.hpp"

class Event {
public:
  virtual void record(cudaStream_t stream) = 0;
  virtual bool query() = 0;
};

class CudaEvent : public Event {
public:
  CudaEvent() {
    cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  }
  ~CudaEvent() {
    cudaEventDestroy(event);
  }
  void record(cudaStream_t stream) override {
    cudaEventRecord(event, stream);
  }
  bool query() override {
    return cudaEventQuery(event) == cudaSuccess;
  }
private:
  cudaEvent_t event;
};

class CustomEvent : public Event {
public:
  CustomEvent() {
    cudaMallocHost(&event, sizeof(int32_t));
    __atomic_store_n(event, 1, __ATOMIC_SEQ_CST);
    cuMemHostGetDevicePointer(
      &dev_ptr, event, 0);
  }
  ~CustomEvent() {
    cudaFreeHost(event);
  }
  void record(cudaStream_t stream) override {
    __atomic_store_n(event, 0, __ATOMIC_SEQ_CST);
    cuStreamWriteValue32(
      stream, dev_ptr, 1, CU_STREAM_WRITE_VALUE_DEFAULT);
  }
  bool query() override {
    return __atomic_load_n(event, __ATOMIC_SEQ_CST);
  }
private:
  int32_t* event __attribute__((aligned(64)));
  CUdeviceptr dev_ptr;
};

void do_benchmark(cudaStream_t stream, Event& event) {
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
    cudaStreamSynchronize(stream);
  }
  std::cout << "Launch: " << SummaryStats(launch_times) << std::endl;
  std::cout << "Query: " << SummaryStats(times) << std::endl;
}

int main(int, char**) {
  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  {
    CudaEvent cuda_event;
    CustomEvent custom_event;
    std::cout << "Custom event:" << std::endl;
    do_benchmark(stream, custom_event);
    std::cout << "CUDA Event:" << std::endl;
    do_benchmark(stream, cuda_event);
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return 0;
}
