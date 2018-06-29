#include "mpi_cuda_impl.hpp"

namespace Al {

// Initialize this.
cudaEvent_t MPICUDABackend::sync_event = (cudaEvent_t) 0;

namespace internal {
namespace mpi_cuda {

void init(int&, char**&) {
  AL_CHECK_CUDA(cudaEventCreateWithFlags(&MPICUDABackend::sync_event,
                                         cudaEventDisableTiming));
}

void finalize() {
  AL_CHECK_CUDA(cudaEventDestroy(MPICUDABackend::sync_event));
}

}  // namespace mpi_cuda
}  // namespace internal
}  // namespace Al
