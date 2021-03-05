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

#include <vector>
#include <mutex>
#include "Al.hpp"
#include "aluminum/cuda.hpp"
#include "aluminum/mempool.hpp"
#include "aluminum/helper_kernels.hpp"

namespace Al {
namespace internal {
namespace cuda {

namespace {
// Stack of CUDA events for reuse.
std::vector<cudaEvent_t> cuda_events;
// Lock to protect the CUDA events.
std::mutex cuda_events_lock;
// Internal CUDA streams.
constexpr int num_internal_streams = 5;
cudaStream_t internal_streams[num_internal_streams];
// Whether stream memory operations are supported.
bool stream_mem_ops_supported = false;
// Whether we're using external streams (these are not freed).
bool using_external_streams = false;
}

void init(int&, char**&) {
  // Initialize internal streams.
  for (int i = 0; i < num_internal_streams; ++i) {
    // Set highest priority if instructed
    if (std::getenv("AL_USE_PRIORITY_STREAM")) {
      int least_priority, greatest_priority;
      AL_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(
          &least_priority, &greatest_priority));
      AL_CHECK_CUDA(cudaStreamCreateWithPriority(
          &internal_streams[i], cudaStreamDefault, greatest_priority));
    } else {
      AL_CHECK_CUDA(cudaStreamCreate(&internal_streams[i]));
    }
    profiling::name_stream(internal_streams[i],
                           "al_internal_" + std::to_string(i));
  }
#ifndef AL_HAS_ROCM
  // Check whether stream memory operations are supported.
  CUdevice dev;
  AL_CHECK_CUDA_DRV(cuCtxGetDevice(&dev));
  int attr;
  AL_CHECK_CUDA_DRV(cuDeviceGetAttribute(
                      &attr, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev));
  stream_mem_ops_supported = attr;
#else
  stream_mem_ops_supported = false;
#endif
  // Preallocate memory for synchronization operations.
  std::vector<int32_t*> prealloc_mem;
  for (int i = 0; i < AL_SYNC_MEM_PREALLOC; ++i) {
    prealloc_mem.push_back(get_pinned_memory<int32_t>(1));
  }
  for (int i = 0; i < AL_SYNC_MEM_PREALLOC; ++i) {
    release_pinned_memory(prealloc_mem[i]);
  }
}

void finalize() {
  for (auto&& event : cuda_events) {
    AL_CHECK_CUDA(cudaEventDestroy(event));
  }
  if (!using_external_streams) {
    for (int i = 0; i < num_internal_streams; ++i) {
      AL_CHECK_CUDA(cudaStreamDestroy(internal_streams[i]));
    }
  }
}

cudaEvent_t get_cuda_event() {
  std::lock_guard<std::mutex> lock(cuda_events_lock);
  cudaEvent_t event;
  if (cuda_events.empty()) {
    AL_CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  } else {
    event = cuda_events.back();
    cuda_events.pop_back();
  }
  return event;
}

void release_cuda_event(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(cuda_events_lock);
  cuda_events.push_back(event);
}

cudaStream_t get_internal_stream() {
  static size_t cur_stream = 0;
  return internal_streams[cur_stream++ % num_internal_streams];
}

cudaStream_t get_internal_stream(size_t id) {
  return internal_streams[id];
}

void replace_internal_streams(std::function<cudaStream_t()> stream_getter) {
  // Clean up our streams if needed.
  if (!using_external_streams) {
    for (int i = 0; i < num_internal_streams; ++i) {
      AL_CHECK_CUDA(cudaStreamDestroy(internal_streams[i]));
    }
  }
  for (int i = 0; i < num_internal_streams; ++i) {
    internal_streams[i] = stream_getter();
  }
  using_external_streams = true;
}

bool stream_memory_operations_supported() {
  return stream_mem_ops_supported;
}

FastEvent::FastEvent() {
  if (stream_memory_operations_supported()) {
    sync_event = get_pinned_memory<int32_t>(1);
    // Initialize to completed to match CUDA event semantics.
    __atomic_store_n(sync_event, 1, __ATOMIC_SEQ_CST);
    AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(
                        &sync_event_dev_ptr, sync_event, 0));
  } else {
    plain_event = get_cuda_event();
  }
}

FastEvent::~FastEvent() {
  if (stream_memory_operations_supported()) {
    release_pinned_memory(sync_event);
  } else {
    release_cuda_event(plain_event);
  }
}

void FastEvent::record(cudaStream_t stream) {
  if (stream_memory_operations_supported()) {
    // We cannot use std::atomic because we need the actual address of
    // the memory.
#ifndef AL_HAS_ROCM
    __atomic_store_n(sync_event, 0, __ATOMIC_SEQ_CST);
    AL_CHECK_CUDA_DRV(cuStreamWriteValue32(
                        stream, sync_event_dev_ptr, 1,
                        CU_STREAM_WRITE_VALUE_DEFAULT));
#else
    throw_al_exception("A serious error has occurred; should not reach this.");
#endif
  } else {
    AL_CHECK_CUDA(cudaEventRecord(plain_event, stream));
  }
}

bool FastEvent::query() {
  if (stream_memory_operations_supported()) {
    return __atomic_load_n(sync_event, __ATOMIC_SEQ_CST);
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

GPUWait::GPUWait()
  : wait_sync(get_pinned_memory<int32_t>(1)),
    wait_sync_dev_ptr_no_stream_mem_ops(nullptr),
    wait_sync_dev_ptr(0U)
{
  // An atomic here may be overkill.
  // Can't use std::atomic because we need the actual address.
  __atomic_store_n(wait_sync, 0, __ATOMIC_SEQ_CST);

  if (stream_memory_operations_supported())
    AL_CHECK_CUDA_DRV(cuMemHostGetDevicePointer(
                        &wait_sync_dev_ptr, wait_sync, 0));
  else
    AL_CHECK_CUDA(
      cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&wait_sync_dev_ptr_no_stream_mem_ops),
        wait_sync,
        0));
}

GPUWait::~GPUWait() {
  release_pinned_memory(wait_sync);
}

void GPUWait::wait(cudaStream_t stream) {
  if (stream_memory_operations_supported())
#ifdef AL_HAS_ROCM
    launch_wait_kernel(stream, 1, static_cast<int32_t*>(wait_sync_dev_ptr));
#else
    launch_wait_kernel(stream, 1, wait_sync_dev_ptr);
#endif
    else
    launch_wait_kernel(stream, 1, wait_sync_dev_ptr_no_stream_mem_ops);
}

void GPUWait::signal() {
  __atomic_store_n(wait_sync, 1, __ATOMIC_SEQ_CST);
}

}  // namespace cuda
}  // namespace internal
}  // namespace Al
