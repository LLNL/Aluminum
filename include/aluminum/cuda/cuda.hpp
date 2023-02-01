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
#include "aluminum/base.hpp"

#include <sstream>
#include <string>

#if defined AL_HAS_ROCM
#include <hip/hip_runtime.h>
#elif defined AL_HAS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "aluminum/datatypes.hpp"

#ifdef AL_HAS_ROCM
// HIP has no equivalent for the driver version of this.
inline hipError_t cuGetErrorString(hipError_t const error, const char** pStr)
{
  *pStr = hipGetErrorString(error);
  return hipSuccess;
}
#endif // AL_HAS_ROCM

// Note: These macros only work inside the Al namespace.
#define AL_GPU_SYNC(async)                                              \
  do {                                                                  \
    /* Synchronize GPU and check for errors. */                         \
    auto status_GPU_SYNC = AlGpuDeviceSynchronize();              \
    if (status_GPU_SYNC != AlGpuSuccess) {                        \
      std::ostringstream err_GPU_SYNC;                                  \
      if (async) { err_GPU_SYNC << "Asynchronous "; }                   \
      err_GPU_SYNC << "GPU error: "                                     \
                   << AlGpuGetErrorString(status_GPU_SYNC);       \
      throw_al_exception(err_GPU_SYNC.str());                           \
    }                                                                   \
  } while (0)

#define AL_FORCE_CHECK_GPU(gpu_rt_call)                         \
  do {                                                          \
    /* Call GPU API routine, synchronizing before and */        \
    /* after to check for errors. */                            \
    AL_GPU_SYNC(true);                                          \
    auto status_CHECK_GPU = (gpu_rt_call);                      \
    if (status_CHECK_GPU != AlGpuSuccess) {               \
      throw_al_exception(                                       \
        std::string("GPU error: ")                              \
        + AlGpuGetErrorString(status_CHECK_GPU));         \
    }                                                           \
    AL_GPU_SYNC(false);                                         \
  } while (0)

#define AL_FORCE_CHECK_GPU_NOSYNC(gpu_rt_call)          \
  do {                                                  \
    auto status_CHECK_GPU = (gpu_rt_call);              \
    if (status_CHECK_GPU != AlGpuSuccess) {       \
      throw_al_exception(                               \
        std::string("GPU error: ")                      \
        + AlGpuGetErrorString(status_CHECK_GPU)); \
    }                                                   \
  } while (0)

#define AL_FORCE_CHECK_GPU_DRV(gpu_drv_call)                    \
  do {                                                          \
    AL_GPU_SYNC(true);                                          \
    auto status_CHECK_GPU_DRV = (gpu_drv_call);                 \
    if (status_CHECK_GPU_DRV != AL_GPU_DRV_SUCCESS) {           \
      const char* err_msg_CHECK_GPU_DRV;                        \
      AL_IGNORE_NODISCARD(                                      \
        cuGetErrorString(status_CHECK_GPU_DRV,                  \
                         &err_msg_CHECK_GPU_DRV));              \
      throw_al_exception(std::string("GPU driver error: ")      \
                         + err_msg_CHECK_GPU_DRV);              \
    }                                                           \
    AL_GPU_SYNC(false);                                         \
  } while (0)

#define AL_FORCE_CHECK_GPU_DRV_NOSYNC(gpu_drv_call)             \
  do {                                                          \
    auto status_CHECK_GPU_DRV = (gpu_drv_call);                 \
    if (status_CHECK_GPU_DRV != AL_GPU_DRV_SUCCESS) {           \
      const char* err_msg_CHECK_GPU_DRV;                        \
      AL_IGNORE_NODISCARD(                                      \
        cuGetErrorString(status_CHECK_GPU_DRV,                  \
                         &err_msg_CHECK_GPU_DRV));              \
      throw_al_exception(std::string("GPU driver error: ")      \
                         + err_msg_CHECK_GPU_DRV);              \
    }                                                           \
  } while (0)

// NOTE: These are used throughout the code, so it'd be a LARGE diff
// to update them to "AL_CHECK_GPU_..."
#define AL_CHECK_CUDA(cuda_call) AL_FORCE_CHECK_GPU_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_NOSYNC(cuda_call) AL_FORCE_CHECK_GPU_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_DRV(cuda_call) AL_FORCE_CHECK_GPU_DRV_NOSYNC(cuda_call)
#define AL_CHECK_CUDA_DRV_NOSYNC(cuda_call) \
  AL_FORCE_CHECK_GPU_DRV_NOSYNC(cuda_call)

namespace Al {
namespace internal {
namespace cuda {

/** Do GPU initialization. */
void init(int& argc, char**& argv);
/** Finalize GPU. */
void finalize();

/** Return whether stream memory operations are supported. */
bool stream_memory_operations_supported();

}  // namespace cuda
}  // namespace internal
}  // namespace Al
