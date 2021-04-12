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

#include <cuda_runtime.h>
#include "aluminum/cuda/helper_kernels.hpp"

namespace Al {
namespace internal {
namespace cuda {

__global__ void spin_wait_kernel(int32_t wait_value, volatile int32_t* wait_mem) {
  for (;;)
  {
    __threadfence_system();
    int32_t value = *wait_mem;
    if (value == wait_value) break;
  }
}

void launch_wait_kernel(cudaStream_t stream, int32_t wait_value, volatile int32_t* wait_mem) {
  spin_wait_kernel<<<1,1,0,stream>>>(wait_value, wait_mem);
}

#if defined AL_HAS_CUDA && !defined AL_HAS_ROCM
void launch_wait_kernel(cudaStream_t stream, int32_t wait_value, CUdeviceptr wait_mem) {
  AL_CHECK_CUDA_DRV(cuStreamWaitValue32(
                      stream, wait_mem, wait_value, CU_STREAM_WAIT_VALUE_EQ));
}
#endif // defined AL_HAS_CUDA && !defined AL_HAS_ROCM

} // namespace cuda
} // namespace internal
} // namespace Al
