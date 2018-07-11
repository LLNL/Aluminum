#include <vector>
#include "cuda.hpp"

namespace Al {
namespace internal {
namespace cuda {

namespace {
// Stack of CUDA events for reuse.
std::vector<cudaEvent_t> cuda_events;
// Internal CUDA streams.
constexpr int num_internal_streams = 5;
cudaStream_t internal_streams[num_internal_streams];
}

void init(int&, char**&) {
  // Initialize internal streams.
  for (int i = 0; i < num_internal_streams; ++i) {
    AL_CHECK_CUDA(cudaStreamCreate(&internal_streams[i]));
  }
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
  cuda_events.push_back(event);
}

cudaStream_t get_internal_stream() {
  static size_t cur_stream = 0;
  return internal_streams[cur_stream++ % num_internal_streams];
}

cudaStream_t get_internal_stream(size_t id) {
  return internal_streams[id];
}

}  // namespace cuda
}  // namespace internal
}  // namespace Al
