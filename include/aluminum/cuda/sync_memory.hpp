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

#include <cstdint>
#include <cstdlib>
#include "aluminum/utils/locked_resource_pool.hpp"
#include "aluminum/cuda/cuda.hpp"
#include "aluminum/tuning_params.hpp"

namespace Al {
namespace internal {
namespace cuda {

// TODO: May want to allocate larger chunks and partition.

/**
 * Allocate CUDA pinned memory such that there is one allocation per
 * cache line.
 */
struct CacheLinePinnedMemoryAllocator {
  int32_t *allocate() {
    // Overallocate to avoid interference.
    int32_t *mem = (int32_t *)std::aligned_alloc(
        AL_DESTRUCTIVE_INTERFERENCE_SIZE, AL_DESTRUCTIVE_INTERFERENCE_SIZE);
    AL_CHECK_CUDA(cudaHostRegister(mem, AL_DESTRUCTIVE_INTERFERENCE_SIZE,
                                   cudaHostRegisterDefault));
    return mem;
  }

  void deallocate(int32_t* mem) {
    AL_CHECK_CUDA(cudaHostUnregister(mem));
    std::free(mem);
  }
};

/** Resource pool for synchronization memory. */
extern Al::internal::LockedResourcePool<int32_t*, CacheLinePinnedMemoryAllocator> sync_pool;

}  // namespace cuda
}  // namespace internal
}  // namespace Al
