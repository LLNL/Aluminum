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

#include <memory>

#include "aluminum/cuda/cuda.hpp"

// We forward-declare CUB here so we can define a pointer to the class.
#if defined AL_HAS_ROCM
#define AL_CUB_NS hipcub
#elif defined AL_HAS_CUDA
#define AL_CUB_NS cub
#endif
namespace AL_CUB_NS {
class CachingDeviceAllocator;
}

namespace Al {
namespace internal {

/**
 * This is a shim around (Hip)CUB to prevent it needing to be visible
 * to external users of LBANN.
 */
class DeviceMempoolShim {
public:
  DeviceMempoolShim(unsigned int bin_growth);
  ~DeviceMempoolShim();
  void allocate(void** ptr, size_t size, AlGpuStream_t stream);
  void release(void* ptr);
  void clear();
private:
  std::unique_ptr<AL_CUB_NS::CachingDeviceAllocator> cub_pool;
};

}  // namespace internal
}  // namespace Al
