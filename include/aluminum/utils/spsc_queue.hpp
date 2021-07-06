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

#include <atomic>
#include <algorithm>
#include <type_traits>
#include "aluminum/base.hpp"
#include "aluminum/tuning_params.hpp"
#include "aluminum/utils/meta.hpp"

namespace Al {
namespace internal {

/**
 * Bounded, lock-free, single-producer, single-consumer queue.
 *
 * This is Lamport's classic SPSC queue with memory order optimizations.
 * See Le, et al. "Correct and Efficient Bounded FIFO Queues".
 */
template <typename T>
class SPSCQueue {
public:
  /** Initialize queue with fixed size (must be a power of 2). */
  explicit SPSCQueue(size_t size_)
#ifndef AL_DEBUG
    noexcept
#endif
    : size(size_), front(0), back(0) {
    static_assert(std::is_pointer<T>::value, "T must be a pointer type");
#ifdef AL_DEBUG
    if (!is_pow2(size)) {
      throw_al_exception("SPSCQueue size must be a power of 2");
    }
#endif
    data = new T[size];
    std::fill_n(data, size, nullptr);
  }

  ~SPSCQueue() {
    delete[] data;
  }

  /** Add v to the queue. */
  void push(T& v)
#ifndef AL_DEBUG
    noexcept
#endif
  {
    size_t b = back.load(std::memory_order_relaxed);
    size_t bmod = (b+1) & (size-1);
#ifdef AL_DEBUG
    size_t f = front.load(std::memory_order_acquire);
    if (bmod == f) {
      throw_al_exception("Queue full");
    }
#endif
    data[b] = v;
    back.store(bmod, std::memory_order_release);
  }

  /** Return the next element in the queue; nullptr if empty. */
  T pop() noexcept {
    size_t f = front.load(std::memory_order_relaxed);
    size_t b = back.load(std::memory_order_acquire);
    if (b == f) {
      return nullptr;
    }
    T* v = data[f];
    front.store((f+1) & (size-1), std::memory_order_release);
    return v;
  }

  /**
   * Discard the element at the front of the queue.
   *
   * It is an error to call this if no element is present.
   */
  void pop_always()
#ifndef AL_DEBUG
    noexcept
#endif
  {
    size_t f = front.load(std::memory_order_relaxed);
#ifdef AL_DEBUG
    size_t b = back.load(std::memory_order_acquire);
    if (b == f) {
      throw_al_exception("Tried to pop_always when empty");
    }
#endif
    front.store((f+1) & (size-1), std::memory_order_release);
  }

  /** Return the next element in the queue; nullptr if empty. */
  T peek() noexcept {
    size_t f = front.load(std::memory_order_relaxed);
    size_t b = back.load(std::memory_order_acquire);
    if (b == f) {
      return nullptr;
    }
    return data[f];
  }

private:
  /** Number of elements the queue can store. */
  const size_t size;
  /** Buffer for data in the queue. */
  T* data;
  /** Index for the current front of the queue. */
  alignas(AL_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t> front;
  /** Index for the current back of the queue. */
  alignas(AL_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t> back;

  // Prevent allocations on the cache line back is in.
  char padding[AL_DESTRUCTIVE_INTERFERENCE_SIZE - sizeof(std::atomic<size_t>)];
};

}  // namespace internal
}  // namespace Al
