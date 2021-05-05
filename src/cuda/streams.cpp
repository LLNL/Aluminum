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

#include "aluminum/cuda/streams.hpp"
#include "aluminum/profiling.hpp"

namespace Al {
namespace internal {
namespace cuda {

StreamPool stream_pool = StreamPool(0);

StreamPool::StreamPool(size_t num_streams) {
  allocate(num_streams);
}

StreamPool::~StreamPool() {
  clear();
}

void StreamPool::allocate(size_t num_streams) {
  if (num_streams == 0) {
    return;
  }
  if (!default_streams.empty() || !high_priority_streams.empty()) {
    throw_al_exception("Cannot reallocate existing streams");
  }
  int least_priority, greatest_priority;
  AL_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(
                  &least_priority, &greatest_priority));
  default_streams.resize(num_streams);
  high_priority_streams.resize(num_streams);
  for (size_t i = 0; i < num_streams; ++i) {
    AL_CHECK_CUDA(cudaStreamCreate(&default_streams[i]));
    // TODO: Support stream pool names to better differentiate.
    profiling::name_stream(default_streams[i],
                           "al_internal_" + std::to_string(i));
    AL_CHECK_CUDA(cudaStreamCreateWithPriority(
          &high_priority_streams[i], cudaStreamDefault, greatest_priority));
    profiling::name_stream(high_priority_streams[i],
                           "al_internal_high_" + std::to_string(i));
  }
}

void StreamPool::clear() {
  if (!external_streams) {
    for (size_t i = 0; i < default_streams.size(); ++i) {
      AL_CHECK_CUDA(cudaStreamDestroy(default_streams[i]));
      AL_CHECK_CUDA(cudaStreamDestroy(high_priority_streams[i]));
    }
  }
  default_streams.clear();
  high_priority_streams.clear();
  default_idx = 0;
  high_priority_idx = 0;
}

cudaStream_t StreamPool::get_stream() {
#ifdef AL_DEBUG
  if (default_streams.empty()) {
    throw_al_exception("No default priority streams in pool");
  }
#endif
  uint32_t idx = (default_idx++) % default_streams.size();
  return default_streams[idx];
}

cudaStream_t StreamPool::get_high_priority_stream() {
#ifdef AL_DEBUG
  if (high_priority_streams.empty()) {
    throw_al_exception("No high priority streams in pool");
  }
#endif
  uint32_t idx = (high_priority_idx++) % high_priority_streams.size();
  return high_priority_streams[idx];
}

void StreamPool::replace_streams(std::function<cudaStream_t(bool)> stream_getter) {
  size_t num_streams = default_streams.size();
  // Clean up our streams if needed.
  clear();
  default_streams.resize(num_streams);
  high_priority_streams.resize(num_streams);
  for (size_t i = 0; i < num_streams; ++i) {
    default_streams[i] = stream_getter(false);
    high_priority_streams[i] = stream_getter(true);
  }
  external_streams = true;
}

}  // namespace cuda
}  // namespace internal
}  // namespace Al
