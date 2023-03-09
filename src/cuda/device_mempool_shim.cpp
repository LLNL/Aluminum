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

#include "aluminum/cuda/device_mempool_shim.hpp"

#if defined AL_HAS_ROCM
#include <hipcub/hipcub.hpp>
#elif defined AL_HAS_CUDA
#include <cub/util_allocator.cuh>
#endif


namespace Al {
namespace internal {

DeviceMempoolShim::DeviceMempoolShim(unsigned int bin_growth) {
  cub_pool = std::make_unique<AL_CUB_NS::CachingDeviceAllocator>(bin_growth);
}

// This must be defined here, as the CachingDeviceAllocator is incomplete
// in the header and unique_ptr cannot define its deleter.
DeviceMempoolShim::~DeviceMempoolShim() = default;

void DeviceMempoolShim::allocate(void** ptr, size_t size, AlGpuStream_t stream) {
  AL_CHECK_CUDA(cub_pool->DeviceAllocate(ptr, size, stream));
}

void DeviceMempoolShim::release(void* ptr) {
  AL_CHECK_CUDA(cub_pool->DeviceFree(ptr));
}

void DeviceMempoolShim::clear() {
  AL_IGNORE_NODISCARD(cub_pool->FreeAllCached());
}

}  // namespace internal
}  // namespace Al
