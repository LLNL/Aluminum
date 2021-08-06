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
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "aluminum/base.hpp"

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
      AL_IGNORE_NODISCARD(                                      \
        cuGetErrorString(status_CHECK_CUDA_DRV,                 \
                         &err_msg_CHECK_CUDA_DRV));             \
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
      AL_IGNORE_NODISCARD(                                      \
        cuGetErrorString(status_CHECK_CUDA_DRV,                 \
                         &err_msg_CHECK_CUDA_DRV));             \
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

/** Return whether stream memory operations are supported. */
bool stream_memory_operations_supported();

}  // namespace cuda
}  // namespace internal
}  // namespace Al
