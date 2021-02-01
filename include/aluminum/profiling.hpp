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

#include <thread>
#include <string>

#include <Al_config.hpp>

#ifdef AL_HAS_NVPROF
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif
#ifdef AL_HAS_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace Al {
namespace internal {
namespace profiling {

/** Assign a name to the thread given by handle. */
void name_thread(std::thread::native_handle_type handle, std::string name);
#ifdef AL_HAS_CUDA
/** Assign a name to a CUDA stream. */
void name_stream(cudaStream_t stream, std::string name);
#endif

/** Create an instantaneous marker. */
void mark(std::string desc);

/** Represent a range for profiling. */
struct ProfileRange {
#ifdef AL_HAS_NVPROF
  nvtxRangeId_t nvtx_range;
#endif
};

/** Start a profiling region with name. */
ProfileRange prof_start(std::string name);
/** End a profiling region. */
void prof_end(ProfileRange range);

}  // namespace profiling
}  // namespace internal
}  // namespace Al
