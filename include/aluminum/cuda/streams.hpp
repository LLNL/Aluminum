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

#include <atomic>
#include <vector>
#include <functional>
#include "aluminum/cuda/cuda.hpp"

namespace Al {
namespace internal {
namespace cuda {

/**
 * Manages a set of CUDA streams, accessed in round-robin order.
 *
 * Streams are either default priority or high priority.
 *
 * It is safe for multiple threads to call get_stream concurrently.
 */
class StreamPool {
public:
  /** Create pool with num_streams default and high priority streams. */
  StreamPool(size_t num_streams = 0);
  ~StreamPool();

  /** Explicitly allocate streams. */
  void allocate(size_t num_streams);

  /** Delete all streams in the pool. */
  void clear();

  /** Return a default-priority CUDA stream. */
  cudaStream_t get_stream();

  /**
   * Return a high-priority CUDA stream.
   *
   * If high-priority streams are not supported, returns a default-priority
   * stream.
   */
  cudaStream_t get_high_priority_stream();

  /**
   * Replace all streams in the pool with streams from an external source.
   *
   * Streams provided this way will not be freed by Aluminum.
   *
   * @param stream_getter Return the next stream to use in the pool. This
   * may be called an arbitrary number of times. It takes a boolean argument
   * for whether to return a default (false) or high (true) priority stream.
   */
  void replace_streams(std::function<cudaStream_t(bool)> stream_getter);

private:
  std::vector<cudaStream_t> default_streams;
  std::atomic<uint32_t> default_idx{0};
  std::vector<cudaStream_t> high_priority_streams;
  std::atomic<uint32_t> high_priority_idx{0};
  /** Whether streams were replaced; we do not free these streams. */
  bool external_streams = false;
};

/** Default internal stream pool for Aluminum. */
extern StreamPool stream_pool;

}  // namespace cuda
}  // namespace internal
}  // namespace Al
