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
 * Have a GPU stream block until signalled.
 * This essentially uses full/empty bit semantics to implement synchronization.
 * The GPU will wait on a memory location until the host writes to it using the
 * stream memory wait operation.
 *
 * If stream memory operations are not available, this will use a
 * spinning wait kernel. This can cause problems. It has a tendency to
 * lead to deadlock, especially in "debug" mode. Also, if kernel
 * timeout is enabled, this is likely to error out.
 */
class GPUWait {
 public:
  GPUWait();
  ~GPUWait();
  /** Enqueue a wait onto stream. */
  void wait(cudaStream_t stream);
  /** Signal the stream to continue. */
  void signal();
 private:
  int32_t* wait_sync __attribute__((aligned(64)));
  union {
    int32_t *wait_sync_dev_ptr_no_stream_mem_ops __attribute__((aligned(64)));
    CUdeviceptr wait_sync_dev_ptr;
  };
};

}  // namespace cuda
}  // namespace internal
}  // namespace Al
