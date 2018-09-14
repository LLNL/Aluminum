#pragma once

#include <mutex>
#include <vector>
#include <unordered_map>
#include <utility>
#ifdef AL_HAS_CUDA
#include "cuda.hpp"
#endif

namespace Al {
namespace internal {

/** An entry in a simple memory pool. */
template <typename T>
using MempoolEntry = std::pair<T*, bool>;
/** A list for a particular size of allocation. */
template <typename T>
using MempoolSizeList = std::vector<MempoolEntry<T>>;
/** For actually mapping sizes to entries. */
template <typename T>
using MempoolMap = std::unordered_map<size_t, MempoolSizeList<T>>;
/** A memory pool. */
template <typename T>
struct Mempool {
  /** The actual memory. */
  MempoolMap<T> memmap;
  /** Used to find allocated memory for freeing. */
  std::unordered_map<T*, size_t> allocated;
#if AL_LOCK_MEMPOOL
  /** Mutex to protect concurrent access. */
  std::mutex lock;
#endif
};
/** Get the memory pool for type T. */
template <typename T>
Mempool<T>& get_mempool() {
  static Mempool<T> mempool;
  return mempool;
}

/** Get memory of type T with count elements. */
template <typename T>
T* get_memory(size_t count) {
  auto& pool = get_mempool<T>();
#if AL_LOCK_MEMPOOL
  std::lock_guard<std::mutex> lock(pool.lock);
#endif
  // Try to find free memory.
  if (pool.memmap.count(count)) {
    // See if any memory of this size is free.
    for (auto&& entry : pool.memmap[count]) {
      if (!entry.second) {
        entry.second = true;
        pool.allocated[entry.first] = count;
        return entry.first;
      }
    }
    // No memory free, allocate some.
    T* mem = new T[count];
    pool.memmap[count].emplace_back(mem, true);
    pool.allocated[mem] = count;
    return mem;
  } else {
    // No entry for this size; so no free memory.
    pool.memmap.emplace(count, MempoolSizeList<T>());
    T* mem = new T[count];
    pool.memmap[count].emplace_back(mem, true);
    pool.allocated[mem] = count;
    return mem;
  }
}

/** Release memory that you got with get_memory. */
template <typename T>
void release_memory(T* mem) {
  auto& pool = get_mempool<T>();
#if AL_LOCK_MEMPOOL
  std::lock_guard<std::mutex> lock(pool.lock);
#endif
  size_t count = pool.allocated[mem];
  pool.allocated.erase(mem);
  // Find the entry.
  for (auto&& entry : pool.memmap[count]) {
    if (entry.first == mem) {
      entry.second = false;
      return;
    }
  }
}

#ifdef AL_HAS_CUDA
// Separate pools for CUDA-registered pinned memory.
// It would be nice to unify these pools.

/** Get the pinned memory pool for type T. */
template <typename T>
Mempool<T>& get_pinned_mempool() {
  static Mempool<T> mempool;
  return mempool;
}

/** Get pinned memory of type T. */
template <typename T>
T* get_pinned_memory(size_t count) {
  auto& pool = get_pinned_mempool<T>();
#if AL_LOCK_MEMPOOL
  std::lock_guard<std::mutex> lock(pool.lock);
#endif
  // Try to find free memory.
  if (pool.memmap.count(count)) {
    // See if any memory of this size is free.
    for (auto&& entry : pool.memmap[count]) {
      if (!entry.second) {
        entry.second = true;
        pool.allocated[entry.first] = count;
        return entry.first;
      }
    }
    // No memory free, allocate some.
    // TODO: Error checking for memory allocation.
    T* mem;
    cudaMallocHost(&mem, count*sizeof(T));
    pool.memmap[count].emplace_back(mem, true);
    pool.allocated[mem] = count;
    return mem;
  } else {
    // No entry for this size; so no free memory.
    pool.memmap.emplace(count, MempoolSizeList<T>());
    T* mem;
    cudaMallocHost(&mem, count*sizeof(T));
    pool.memmap[count].emplace_back(mem, true);
    pool.allocated[mem] = count;
    return mem;
  }
}

/** Release memory that you got with get_pinned_memory. */
template <typename T>
void release_pinned_memory(T* mem) {
  auto& pool = get_pinned_mempool<T>();
#if AL_LOCK_MEMPOOL
  std::lock_guard<std::mutex> lock(pool.lock);
#endif
  size_t count = pool.allocated[mem];
  pool.allocated.erase(mem);
  // Find the entry.
  for (auto&& entry : pool.memmap[count]) {
    if (entry.first == mem) {
      entry.second = false;
      return;
    }
  }
}

#endif  // AL_HAS_CUDA

}  // namespace internal
}  // namespace Al
