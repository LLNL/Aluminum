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

#include <stddef.h>
#include <mutex>
#include <vector>

namespace Al {
namespace internal {

/**
 * Provides thread-safe access to a set of identical resources.
 *
 * These resources could be things like a fixed-size chunk of memory.
 *
 * ResourceAllocator must provide allocate and deallocate methods to
 * create and destroy (resp.) the managed resource (T).
 *
 * LockedResourcePool will guarantee that each instance of the
 * ResourceAllocator will be accessed by only a single thread at a
 * time. If additional locking is necessary for correctness, the
 * ResourceAllocator must provide it.
 */
template <typename T, typename ResourceAllocator>
class LockedResourcePool {
public:
  /** Initialize the resource pool. */
  LockedResourcePool(){};

  ~LockedResourcePool() {
    clear();
  }

  /** Preallocate this many instances of the resource. */
  void preallocate(size_t prealloc) {
    for (size_t i = 0; i < prealloc; ++i) {
      resources.push_back(allocator.allocate());
    }
  }

  /** Get one instance of the resource. */
  T get() {
    std::lock_guard<std::mutex> lg(lock);
    if (resources.empty()) {
      return allocator.allocate();
    } else {
      T resource = resources.back();
      resources.pop_back();
      return resource;
    }
  }

  /** Return an instance of the resource to the pool. */
  void release(T resource) {
    std::lock_guard<std::mutex> lg(lock);
    resources.push_back(resource);
  }

  /** Clear all instances left in the pool. */
  void clear() {
    std::lock_guard<std::mutex> lg(lock);
    for (auto&& resource : resources) {
      allocator.deallocate(resource);
    }
    resources.clear();
  }

private:
  /** Protects access to allocator and resource. */
  std::mutex lock;
  /** Allocator for the resource. */
  ResourceAllocator allocator;
  /** Currently available resources. */
  std::vector<T> resources;
};

}  // namespace internal
}  // namespace Al
