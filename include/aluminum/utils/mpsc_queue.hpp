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
#include <memory>
#include <type_traits>
#include "aluminum/base.hpp"
#include "aluminum/tuning_params.hpp"
#include "aluminum/utils/meta.hpp"

namespace Al {
namespace internal {

/** Bounded, lock-free multiple-producer, single-consumer queue. */
template <typename T>
class MPSCQueue {
public:
  /**
   * Initialize queue with fixed size (must be a power of 2).
   *
   * The queue will hold at most size - 1 elements.
   */
  explicit MPSCQueue(size_t size_)
#ifndef AL_DEBUG
    noexcept
#endif
    : size(size_), index(1)
#ifdef AL_DEBUG
    , cur_size(0)
#endif
  {
    static_assert(std::is_pointer<T>::value, "T must be a pointer type");
#ifdef AL_DEBUG
    if (!is_pow2(size)) {
      throw_al_exception("MPSCQueue size must be a power of 2");
    }
#endif
    data = new queue_entry[size + 1];
    head = &data[0];
    head->next = nullptr;
    tail = head;
  }

  ~MPSCQueue() {
    delete[] data;
  }

  /** Add v to the queue. */
  void push(T& v)
#ifndef AL_DEBUG
    noexcept
#endif
  {
    size_t i = index.fetch_add(1);
    queue_entry* entry = &data[i & (size - 1)];
    entry->value = v;
    entry->next = nullptr;
    queue_entry* old_tail;
    queue_entry* old_next;
    while (true) {
#ifdef AL_DEBUG
      if (cur_size.load() + 1 >= size - 1) {
        throw_al_exception("Queue full");
      }
#endif
      old_tail = tail;
      old_next = tail->next;
      if (old_tail == tail) {
        if (old_next != nullptr) {
          // We didn't read the actual tail, help it get updated.
          __atomic_compare_exchange_n(&tail, &old_tail, old_next, true,
                                      __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
        } else {
          // Attempt to add our entry as the node after the current tail.
          if (__atomic_compare_exchange_n(&tail->next, &old_next, entry, true,
                                          __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
            // Update the tail.
            __atomic_compare_exchange_n(&tail, &old_tail, entry, true,
                                        __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#ifdef AL_DEBUG
            ++cur_size;
#endif
            break;
          }
        }
      }
    }
  }

  /** Return the next element in the queue; nullptr if empty. */
  T pop() noexcept {
    if (head->next == nullptr) {
      return nullptr;
    }
    T value = head->next->value;
    head = head->next;
#ifdef AL_DEBUG
    --cur_size;
#endif
    return value;
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
#ifdef AL_DEBUG
    if (head->next == nullptr) {
      throw_al_exception("Tried to pop_always when empty");
    }
#endif
    head = head->next;
#ifdef AL_DEBUG
    --cur_size;
#endif
  }

  /** Return the next element in the queue; nullptr if empty. */
  T peek() noexcept {
    return (head->next == nullptr) ? nullptr : head->next->value;
  }

private:
  /** Stores each entry in the queue, and the next entry in the order. */
  struct queue_entry {
    T value;
    queue_entry* next;
  };

  /** Number of elements the queue can store. */
  const size_t size;
  /** Buffer for entries in the queue. */
  queue_entry* data;
  /** Current index in data. */
  alignas(AL_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t> index;
#ifdef AL_DEBUG
  /** Used for best-effort debugging to detect queue overflows. */
  alignas(AL_DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t> cur_size;
#endif
  /** Pointer to the current head of the queue. */
  alignas(AL_DESTRUCTIVE_INTERFERENCE_SIZE) queue_entry* head;
  /** Pointer to the current tail of the queue. */
  alignas(AL_DESTRUCTIVE_INTERFERENCE_SIZE) queue_entry* tail;

  // Prevent allocations on the cache line tail is in.
  char padding[AL_DESTRUCTIVE_INTERFERENCE_SIZE - sizeof(queue_entry*)];
};

}  // namespace internal
}  // namespace Al
