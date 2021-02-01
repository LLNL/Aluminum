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

#include "aluminum/profiling.hpp"

namespace Al {
namespace internal {
namespace profiling {

void name_thread(std::thread::native_handle_type handle, std::string name) {
#ifdef AL_HAS_NVPROF
  nvtxNameOsThreadA(handle, name.c_str());
#else
  (void) handle;
  (void) name;
#endif
}

#ifdef AL_HAS_CUDA
void name_stream(cudaStream_t stream, std::string name) {
#ifdef AL_HAS_NVPROF
  nvtxNameCudaStreamA(stream, name.c_str());
#else
  (void) stream;
  (void) name;
#endif
}
#endif

void mark(std::string desc) {
#ifdef AL_HAS_NVPROF
  nvtxMarkA(desc.c_str());
#else
  (void) desc;
#endif
}

ProfileRange prof_start(std::string name) {
  ProfileRange range;
#ifdef AL_HAS_NVPROF
  range.nvtx_range = nvtxRangeStartA(name.c_str());
#else
  (void) name;
#endif
  return range;
}

void prof_end(ProfileRange range) {
#ifdef AL_HAS_NVPROF
  nvtxRangeEnd(range.nvtx_range);
#else
  (void) range;
#endif
}

}  // namespace profiling
}  // namespace internal
}  // namespace Al
