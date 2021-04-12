////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Al_config.hpp>

#include <utility>
#include <sstream>
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/**
 * Things we use that hip doesn't support:
 * - cuStreamWaitValue32
 * - cuStreamWriteValue32
 * - cuGetErrorString
 * Of these, the first two are supported by the non-STREAM_MEM_OP code, so we
 * just need to reimplement cuGetErrorString
 */

#ifdef AL_HAS_ROCM
inline hipError_t cuGetErrorString(hipError_t /* error */, const char** pStr)
{
  static char const* unsupported = "hipGetErrorString is unsupported :(";
  *pStr = unsupported;
  return CUDA_SUCCESS;
}
#endif // AL_HAS_ROCM

// Note: These macros only work inside the Al namespace.

#define AL_CUDA_SYNC(async)                                     \
  do {                                                          \
    /* Synchronize GPU and check for errors. */                 \
    cudaError_t status_CUDA_SYNC = cudaDeviceSynchronize();     \
    if (status_CUDA_SYNC != cudaSuccess) {                      \
      std::ostringstream err_CUDA_SYNC;                         \
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

#define AL_FORCE_CHECK_CUDA_NOSYNC(cuda_call)                   \
  do {                                                          \
    cudaError_t status_CHECK_CUDA = (cuda_call);                \
    if (status_CHECK_CUDA != cudaSuccess) {                     \
      throw_al_exception(std::string("CUDA error: ")            \
                  + cudaGetErrorString(status_CHECK_CUDA));     \
    }                                                           \
  } while (0)

#define AL_FORCE_CHECK_CUDA_DRV(cuda_call)                      \
  do {                                                          \
    AL_CUDA_SYNC(true);                                         \
    CUresult status_CHECK_CUDA_DRV = (cuda_call);               \
    if (status_CHECK_CUDA_DRV != CUDA_SUCCESS) {                \
      const char* err_msg_CHECK_CUDA_DRV;                       \
      cuGetErrorString(status_CHECK_CUDA_DRV,                   \
                       &err_msg_CHECK_CUDA_DRV);                \
      throw_al_exception(std::string("CUDA driver error: ")     \
                         + err_msg_CHECK_CUDA_DRV);             \
    }                                                           \
    AL_CUDA_SYNC(false);                                        \
  } while (0)

#define AL_FORCE_CHECK_CUDA_DRV_NOSYNC(cuda_call)               \
  do {                                                          \
    CUresult status_CHECK_CUDA_DRV = (cuda_call);               \
    if (status_CHECK_CUDA_DRV != CUDA_SUCCESS) {                \
      const char* err_msg_CHECK_CUDA_DRV;                       \
      cuGetErrorString(status_CHECK_CUDA_DRV,                   \
                       &err_msg_CHECK_CUDA_DRV);                \
      throw_al_exception(std::string("CUDA driver error: ")     \
                         + err_msg_CHECK_CUDA_DRV);             \
    }                                                           \
  } while (0)

#ifdef AL_DEBUG
#define AL_CHECK_CUDA(cuda_call) AL_FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_NOSYNC(cuda_call) AL_FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_DRV(cuda_call) AL_FORCE_CHECK_CUDA_DRV_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_DRV_NOSYNC(cuda_call) AL_FORCE_CHECK_CUDA_DRV_NOSYNC(cuda_call)
#else
#define AL_CHECK_CUDA(cuda_call) AL_FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_NOSYNC(cuda_call) AL_FORCE_CHECK_CUDA_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_DRV(cuda_call) AL_FORCE_CHECK_CUDA_DRV_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_DRV_NOSYNC(cuda_call) AL_FORCE_CHECK_CUDA_DRV_NOSYNC(cuda_call)
#endif

namespace Al {
namespace internal {
namespace cuda {

/** Do CUDA initialization. */
void init(int& argc, char**& argv);
/** Finalize CUDA. */
void finalize();

/** Return a currently unused CUDA event. */
cudaEvent_t get_cuda_event();
/** Release a finished CUDA event. */
void release_cuda_event(cudaEvent_t event);

/**
 * Return an internal stream to run operations on.
 * This may select among multiple internal streams.
 */
cudaStream_t get_internal_stream();
/** Get a specific internal stream. */
cudaStream_t get_internal_stream(size_t id);
/**
 * Replace the internal stream pool with user-provided streams.
 *
 * stream_getter may be called an arbitrary number of times and should
 * return the streams to use in the pool.
 *
 * This is meant to help interface with external applications that
 * need Aluminum to use their streams for everything.
 */
void replace_internal_streams(std::function<cudaStream_t()> stream_getter);

/** Return whether stream memory operations are supported. */
bool stream_memory_operations_supported();

/**
 * An optimized version of CUDA events.
 * This essentially uses full/empty bit semantics to implement synchronization.
 * A memory location is polled on by the host and written to by the device
 * using the stream memory write operation.
 * This falls back to the usual CUDA events when stream memory operations are
 * not available.
 * @note This is currently always falling back on CUDA events to work around a
 * hang, the underlying cause of which has not been diagnosed.
 */
class FastEvent {
 public:
  /**
   * Allocate the event.
   */
  FastEvent();
  ~FastEvent();
  /** Record the event into stream. */
  void record(cudaStream_t stream);
  /** Return true if the event has completed. */
  bool query();
 private:
  int32_t* sync_event __attribute__((aligned(64)));
  CUdeviceptr sync_event_dev_ptr;
  cudaEvent_t plain_event;
};

/**
 * Have a GPU stream block until signalled.
 * This essentially uses full/empty bit semantics to implement synchronization.
 * The GPU will wait on a memory location until the host writes to it using the
 * stream memory wait operation.
 *
 * If stream memory operations are not available, this will use a
 * spinning wait kernel. This can cause problems. It has a tendency to
 * lead to deadlock, especially in "debug" mode. Also, if kernel
 * timeout is enabled, this is likely to error out.
 */
class GPUWait {
 public:
  GPUWait();
  ~GPUWait();
  /** Enqueue a wait onto stream. */
  void wait(cudaStream_t stream);
  /** Signal the stream to continue. */
  void signal();
 private:
  int32_t* wait_sync __attribute__((aligned(64)));
  int32_t* wait_sync_dev_ptr_no_stream_mem_ops __attribute__((aligned(64)));
  CUdeviceptr wait_sync_dev_ptr;
};

}  // namespace cuda
}  // namespace internal
}  // namespace Al
