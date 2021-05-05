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

#include "Al.hpp"
#include "aluminum/cuda/gpu_wait.hpp"
#include "aluminum/cuda/helper_kernels.hpp"
#include "aluminum/cuda/sync_memory.hpp"

namespace Al {
namespace internal {
namespace cuda {

GPUWait::GPUWait()
  : wait_sync(sync_pool.get())
{
  // An atomic here may be overkill.
  // Can't use std::atomic because we need the actual address.
  __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);

  if (stream_memory_operations_supported()) {
    AL_CHECK_CUDA_DRV(
        cuMemHostGetDevicePointer(&wait_sync_dev_ptr, wait_sync, 0));
  } else {
    AL_CHECK_CUDA(cudaHostGetDevicePointer(
        reinterpret_cast<void **>(&wait_sync_dev_ptr_no_stream_mem_ops),
        wait_sync, 0));
  }
}

GPUWait::~GPUWait() {
  sync_pool.release(wait_sync);
}

void GPUWait::wait(cudaStream_t stream) {
  if (stream_memory_operations_supported()) {
#ifdef AL_HAS_ROCM
    launch_wait_kernel(stream, 1, static_cast<int32_t *>(wait_sync_dev_ptr));
#else
    launch_wait_kernel(stream, 1, wait_sync_dev_ptr);
#endif
  } else {
    launch_wait_kernel(stream, 1, wait_sync_dev_ptr_no_stream_mem_ops);
  }
}

void GPUWait::signal() {
  __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST);
}

}  // namespace cuda
}  // namespace internal
}  // namespace Al
