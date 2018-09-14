#include <vector>
#include <mutex>
#include "Al.hpp"
#include "cuda.hpp"
#include "mempool.hpp"

namespace Al {
namespace internal {
namespace cuda {

namespace {
// Stack of CUDA events for reuse.
std::vector<cudaEvent_t> cuda_events;
// Lock to protect the CUDA events.
std::mutex cuda_events_lock;
// Internal CUDA streams.
constexpr int num_internal_streams = 5;
cudaStream_t internal_streams[num_internal_streams];
// Whether stream memory operations are supported.
bool stream_mem_ops_supported = false;
}

void init(int&, char**&) {
  // Initialize internal streams.
  for (int i = 0; i < num_internal_streams; ++i) {
    AL_CHECK_CUDA(cudaStreamCreate(&internal_streams[i]));
  }
  // Check whether stream memory operations are supported.
  CUdevice dev;
  AL_CHECK_CUDA_DRV(cuCtxGetDevice(&dev));
  int attr;
  AL_CHECK_CUDA_DRV(cuDeviceGetAttribute(
                      &attr, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev));
  stream_mem_ops_supported = attr;
}
void finalize() {
  for (auto&& event : cuda_events) {
    AL_CHECK_CUDA(cudaEventDestroy(event));
  }
  for (int i = 0; i < num_internal_streams; ++i) {
    AL_CHECK_CUDA(cudaStreamDestroy(internal_streams[i]));
  }
}

cudaEvent_t get_cuda_event() {
  std::lock_guard<std::mutex> lock(cuda_events_lock);
  cudaEvent_t event;
  if (cuda_events.empty()) {
    AL_CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  } else {
    event = cuda_events.back();
    cuda_events.pop_back();
  }
  return event;
}

void release_cuda_event(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(cuda_events_lock);
  cuda_events.push_back(event);
}

cudaStream_t get_internal_stream() {
  static size_t cur_stream = 0;
  return internal_streams[cur_stream++ % num_internal_streams];
}

cudaStream_t get_internal_stream(size_t id) {
  return internal_streams[id];
}

bool stream_memory_operations_supported() {
  return stream_mem_ops_supported;
}

FastEvent::FastEvent() {
  if (!stream_memory_operations_supported()) {
    throw_al_exception("FastEvent requires stream memory operations");
  }
  sync_event = get_pinned_memory<int32_t>(1);
  AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(
                      &sync_event_dev_ptr, sync_event, 0));
}

FastEvent::~FastEvent() {
  release_pinned_memory(sync_event);
}

void FastEvent::record(cudaStream_t stream) {
  // We cannot use std::atomic because we need the actual address of the memory.
  __atomic_store_n(sync_event, 0, __ATOMIC_SEQ_CST);
  AL_CHECK_CUDA_DRV(cuStreamWriteValue32(
                      stream, sync_event_dev_ptr, 1,
                      CU_STREAM_WRITE_VALUE_DEFAULT));
}

bool FastEvent::query() {
  return __atomic_load_n(sync_event, __ATOMIC_SEQ_CST);
}

GPUWait::GPUWait() {
  wait_sync = get_pinned_memory<int32_t>(1);
  // An atomic here may be overkill.
  // Can't use std::atomic because we need the actual address.
  __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);
  AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(
                      &wait_sync_dev_ptr, wait_sync, 0));
}

GPUWait::~GPUWait() {
  release_pinned_memory(wait_sync);
}

void GPUWait::wait(cudaStream_t stream) {
  AL_CHECK_CUDA_DRV(cuStreamWaitValue32(
                      stream, wait_sync_dev_ptr, 1, CU_STREAM_WAIT_VALUE_EQ));
}

void GPUWait::signal() {
  __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST);
}

}  // namespace cuda
}  // namespace internal
}  // namespace Al
