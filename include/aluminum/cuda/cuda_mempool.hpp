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

#include "aluminum/utils/caching_allocator.hpp"
#include "aluminum/cuda/cuda.hpp"
#if defined AL_HAS_ROCM
#include <hipcub/hipcub.hpp>
#elif defined AL_HAS_CUDA
#include <cub/util_allocator.cuh>
#endif

namespace Al {
namespace internal {

/** Allocator for pinned host memory. */
struct CUDAPinnedMemoryAllocator {
  void* allocate(size_t bytes) {
    void* ptr;
    AL_CHECK_CUDA(cudaMallocHost(&ptr, bytes));
    return ptr;
  }

  void deallocate(void* ptr) {
    AL_CHECK_CUDA(cudaFreeHost(ptr));
  }
};

/** Specialized caching allocator for CUDA using CUB. */
template <>
class CachingAllocator<MemoryType::CUDA, void, void> {
public:
  CachingAllocator() : cub_pool(2u) {}

  ~CachingAllocator() {
    clear();
  }

  template <typename T>
  T* allocate(size_t size, cudaStream_t stream) {
    T* mem = nullptr;
    AL_CHECK_CUDA(cub_pool.DeviceAllocate(reinterpret_cast<void**>(&mem),
                                          sizeof(T)*size, stream));
    return mem;
  }

  template <typename T>
  void release(T* ptr) {
    AL_CHECK_CUDA(cub_pool.DeviceFree(ptr));
  }

  void clear() { AL_IGNORE_NODISCARD(cub_pool.FreeAllCached()); }

private:
  cub::CachingDeviceAllocator cub_pool;
};

}  // namespace internal
}  // namespace Al
