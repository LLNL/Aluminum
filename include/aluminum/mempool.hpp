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

#include <cstdlib>
#include "aluminum/utils/caching_allocator.hpp"
#ifdef AL_HAS_CUDA
#include "aluminum/cuda/cuda_mempool.hpp"
#endif

namespace Al {
namespace internal {

/** Allocator for standard host memory. */
struct HostMemoryAllocator {
  void* allocate(size_t bytes) {
    void* ptr = std::malloc(bytes);
#ifdef AL_DEBUG
    if (ptr == nullptr) {
      throw_al_exception("Failed to allocate memory");
    }
#endif
    return ptr;
  }

  void deallocate(void* ptr) {
    std::free(ptr);
  }
};


namespace details {

/** Type of the underlying allocator. */
template <MemoryType MemType> struct allocator_type {};
template <> struct allocator_type<MemoryType::HOST> {
  using type = CachingAllocator<MemoryType::HOST, HostMemoryAllocator>;
};
#ifdef AL_HAS_CUDA
template <> struct allocator_type<MemoryType::CUDA_PINNED_HOST> {
  using type = CachingAllocator<MemoryType::CUDA_PINNED_HOST, CUDAPinnedMemoryAllocator>;
};
template <> struct allocator_type<MemoryType::CUDA> {
  using type = CachingAllocator<MemoryType::CUDA, void, void>;
};
#endif

/** Wrap the set of supported memory pools and support duck-typing. */
struct allocator_wrapper {
  typename allocator_type<MemoryType::HOST>::type host_mempool;
#ifdef AL_HAS_CUDA
  typename allocator_type<MemoryType::CUDA_PINNED_HOST>::type cuda_pinned_mempool;
  typename allocator_type<MemoryType::CUDA>::type cuda_mempool;
#endif

  template <MemoryType MemType>
  typename allocator_type<MemType>::type& get_mempool();
};

template <>
inline typename allocator_type<MemoryType::HOST>::type& allocator_wrapper::get_mempool<MemoryType::HOST>() {
  return host_mempool;
}
#ifdef AL_HAS_CUDA
template <>
inline typename allocator_type<MemoryType::CUDA_PINNED_HOST>::type& allocator_wrapper::get_mempool<MemoryType::CUDA_PINNED_HOST>() {
  return cuda_pinned_mempool;
}
template <>
inline typename allocator_type<MemoryType::CUDA>::type& allocator_wrapper::get_mempool<MemoryType::CUDA>() {
  return cuda_mempool;
}
#endif

}  // namespace details


/** Provide a generic interface to different caching memory allocators. */
class MemoryPool {
public:
  ~MemoryPool() {
    clear();
  }

  /** Allocate space for size instances of type T in memory MemType. */
  template <MemoryType MemType, typename T>
  T* allocate(size_t size) {
    auto& mempool = pools.get_mempool<MemType>();
    return mempool.template allocate<T>(size);
  }

  template <MemoryType MemType, typename T, typename Stream>
  T *allocate(size_t size, Stream stream) {
    auto &mempool = pools.get_mempool<MemType>();
    return mempool.template allocate<T>(size, stream);
  }

  /** Return ptr to the cache for reuse. */
  template <MemoryType MemType, typename T>
  void release(T* ptr) {
    auto& mempool = pools.get_mempool<MemType>();
    mempool.template release<T>(ptr);
  }

  /** Release all cached memory. */
  void clear() {
    pools.get_mempool<MemoryType::HOST>().clear();
#ifdef AL_HAS_CUDA
    pools.get_mempool<MemoryType::CUDA_PINNED_HOST>().clear();
    pools.get_mempool<MemoryType::CUDA>().clear();
#endif
  }

private:
  /** Wrapper to duck-type the caching allocator based on the MemoryType. */
  details::allocator_wrapper pools;
};

extern MemoryPool mempool;

}  // namespace internal
}  // namespace Al
