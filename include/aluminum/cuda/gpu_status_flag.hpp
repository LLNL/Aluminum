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

#include "aluminum/cuda/cuda.hpp"

namespace Al {
namespace internal {
namespace cuda {

/**
 * An optimized version of CUDA events that only supports polling from the host.
 * This essentially uses full/empty bit semantics to implement synchronization.
 * A memory location is polled on by the host and written to by the device
 * using the stream memory write operation.
 * This falls back to the usual CUDA events when stream memory operations are
 * not available.
 * @note This is currently always falling back on CUDA events to work around a
 * hang, the underlying cause of which has not been diagnosed.
 */
class GPUStatusFlag {
 public:
  /**
   * Allocate the event.
   */
  GPUStatusFlag();
  ~GPUStatusFlag();
  /** Record the event into stream. */
  void record(cudaStream_t stream);
  /** Return true if the event has completed. */
  bool query();
 private:
  struct stream_mem_t {
    int32_t* sync_event __attribute__((aligned(64)));
    CUdeviceptr sync_event_dev_ptr;
  };
  union {
    stream_mem_t stream_mem;
    cudaEvent_t plain_event;
  };
};

}  // namespace cuda
}  // namespace internal
}  // namespace Al
