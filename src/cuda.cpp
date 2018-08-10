#include <vector>
#include <mutex>
#include "cuda.hpp"

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

}  // namespace cuda
}  // namespace internal
}  // namespace Al
