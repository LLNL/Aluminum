#pragma once

#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>

// Note: These macros only work inside the Al namespace.

#define AL_CUDA_SYNC(async)                                     \
  do {                                                          \
    /* Synchronize GPU and check for errors. */                 \
    cudaError_t status_CUDA_SYNC = cudaDeviceSynchronize();     \
    if (status_CUDA_SYNC == cudaSuccess)                        \
      status_CUDA_SYNC = cudaGetLastError();                    \
    if (status_CUDA_SYNC != cudaSuccess) {                      \
      cudaDeviceReset();                                        \
      std::stringstream err_CUDA_SYNC;                          \
      if (async) { err_CUDA_SYNC << "Asynchronous "; }          \
      err_CUDA_SYNC << "CUDA error: "                           \
                    << cudaGetErrorString(status_CUDA_SYNC);    \
      throw_al_exception(err_CUDA_SYNC.str());                  \
    }                                                           \
  } while (0)

#define AL_FORCE_CHECK_CUDA(cuda_call)                          \
  do {                                                          \
    /* Call CUDA API routine, synchronizing before and */       \
    /* after to check for errors. */                            \
    AL_CUDA_SYNC(true);                                         \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      throw_al_exception(std::string("CUDA error: ")            \
                  + cudaGetErrorString(status_CHECK_CUDA));     \
    }                                                           \
    AL_CUDA_SYNC(false);                                        \
  } while (0)

#ifdef AL_DEBUG
#define AL_CHECK_CUDA(cuda_call) AL_FORCE_CHECK_CUDA(cuda_call)
#else
#define AL_CHECK_CUDA(cuda_call) (cuda_call)
#endif

namespace Al {
namespace internal {
namespace cuda {

/** Do CUDA initialization. */
void init(int& argc, char**& argv);
/** Finalize CUDA. */
void finalize();

// Pool to re-use CUDA events. These are not thread-safe.
// TODO: Figure out if we need to make these thread-safe.
/** Return a currently unused CUDA event. */
cudaEvent_t get_cuda_event();
/** Release a finished CUDA event. */
void release_cuda_event(cudaEvent_t event);

/**
 * Return an internal stream to run operations on.
 * This may select among multiple internal streams.
 */
cudaStream_t get_internal_stream();

}  // namespace cuda
}  // namespace internal
}  // namespace Al
