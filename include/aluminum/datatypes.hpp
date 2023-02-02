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

// This file identifies support for different specialized datatypes and
// provides some basic things for them as needed.

#include <Al_config.hpp>

// IEEE 16 bit floating point (i.e., fp16 or half).

#if defined AL_HAS_ROCM
#include <hip/hip_fp16.h>
#define AL_HAS_HALF 1
#elif defined AL_HAS_CUDA
#include <cuda_fp16.h>
#define AL_HAS_HALF 1
#endif

// Brain floating point 16 (bfloat16).

#if defined AL_HAS_ROCM
#include <hip/hip_bfloat16.h>
#define AL_HAS_BFLOAT 1
using al_bfloat16 = hip_bfloat16;

// Provide these for compatibility with CUDA.
inline al_bfloat16 __float2bfloat16(const float a) {
  return al_bfloat16(a);
}
inline float __bfloat162float(const al_bfloat16 a) {
  return static_cast<float>(a);
}
#elif defined AL_HAS_CUDA
#include <cuda_bf16.h>
#define AL_HAS_BFLOAT 1
using al_bfloat16 = __nv_bfloat16;
#endif
