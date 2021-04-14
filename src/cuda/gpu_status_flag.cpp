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

#include "Al.hpp"
#include "aluminum/cuda/gpu_status_flag.hpp"
#include "aluminum/cuda/sync_memory.hpp"
#include "aluminum/cuda/events.hpp"

namespace Al {
namespace internal {
namespace cuda {

GPUStatusFlag::GPUStatusFlag() {
  if (stream_memory_operations_supported()) {
    stream_mem.sync_event = sync_pool.get();
    // Initialize to completed to match CUDA event semantics.
    __atomic_store_n(stream_mem.sync_event, 1, __ATOMIC_SEQ_CST);
    AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(
                        &stream_mem.sync_event_dev_ptr,
                        stream_mem.sync_event, 0));
  } else {
    plain_event = event_pool.get();
  }
}

GPUStatusFlag::~GPUStatusFlag() {
  if (stream_memory_operations_supported()) {
    sync_pool.release(stream_mem.sync_event);
  } else {
    event_pool.release(plain_event);
  }
}

void GPUStatusFlag::record(cudaStream_t stream) {
  if (stream_memory_operations_supported()) {
    // We cannot use std::atomic because we need the actual address of
    // the memory.
#ifndef AL_HAS_ROCM
    __atomic_store_n(stream_mem.sync_event, 0, __ATOMIC_SEQ_CST);
    AL_CHECK_CUDA_DRV(cuStreamWriteValue32(
                        stream, stream_mem.sync_event_dev_ptr, 1,
                        CU_STREAM_WRITE_VALUE_DEFAULT));
#else
    throw_al_exception("A serious error has occurred; should not reach this.");
#endif
  } else {
    AL_CHECK_CUDA(cudaEventRecord(plain_event, stream));
  }
}

bool GPUStatusFlag::query() {
  if (stream_memory_operations_supported()) {
    return __atomic_load_n(stream_mem.sync_event, __ATOMIC_SEQ_CST);
  } else {
    cudaError_t r = cudaEventQuery(plain_event);
    if (r == cudaSuccess) {
      return true;
    } else if (r != cudaErrorNotReady) {
      AL_CHECK_CUDA(r);
      return false;  // Never reached.
    } else {
      return false;
    }
  }
}

}  // namespace cuda
}  // namespace internal
}  // namespace Al
