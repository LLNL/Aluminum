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

/** Asynchronous progress engine. */

#pragma once

#include <limits>
#include <functional>
#include <list>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <algorithm>
#include <ostream>
#include <array>

namespace Al {

// Forward declaration.
class Communicator;

}  // namespace Al

#include "aluminum/mpi/communicator.hpp"

namespace Al {
namespace internal {
// Forward declaration.
namespace profiling {
struct ProfileRange;
}

/**
 * Request handle for non-blocking operations.
 * The atomic flag is used to check for completion.
 */
using AlRequest = std::shared_ptr<std::atomic<bool>>;
/** Return a free request for use. */
AlRequest get_free_request();
/** Special marker for null requests. */
static constexpr std::nullptr_t NULL_REQUEST = nullptr;
/** Special marker for the default compute stream. */
static constexpr std::nullptr_t DEFAULT_STREAM = nullptr;
/** Run queue types for the progress engine. */
enum class RunType {
  /** Only a limited number of ops will be run at a time. */
  bounded,
  /** There cannot be a limit on how many ops of this type will run. */
  unbounded
};

/** Actions a state can ask the progress engine to do. */
enum class PEAction {
  /** Do nothing (i.e. keep running as it is now). */
  cont,
  /** Advance the state to the next pipeline stage. */
  advance,
  /** Operation is complete. */
  complete
};

/**
 * Represents the state and algorithm for an asynchronous operation.
 * A non-blocking operation should create one of these and enqueue it for
 * execution by the progress thread. Specific implementations can override
 * as needed.
 *
 * An algorithm should be broken up into steps which execute some small,
 * discrete operation. Steps from different operations may be interleaved.
 * Note that the memory pool is not thread-safe, so memory from it should be
 * pre-allocated before enqueueing.
 *
 * The steps are run through a simple pipeline. The algorithm can request it
 * advance to the next stage by returning PEAction::advance. Operations enqueud
 * on the same compute stream will only be advanced in the order they were
 * enqueued. If a state asks to advance but it is not at the head of its
 * pipeline stage, step will not be called again until it has successfully
 * advanced.
 */
class AlState {
  friend class ProgressEngine;
 public:
  /** Create with an associated request. */
  AlState(AlRequest req_) : req(req_) {}
  virtual ~AlState();
  /**
   * Perform initial setup of the algorithm.
   * This is called by the progress engine when the operation begins execution.
   */
  virtual void start();
  /**
   * Run one step of the algorithm.
   * Return the action the algorithm wishes the progress engine to take.
   */
  virtual PEAction step() = 0;
  /** Return the associated request. */
  AlRequest& get_req() { return req; }
  /** True if this is meant to be waited on by the user. */
  virtual bool needs_completion() const { return true; }
  /** Return the compute stream associated with this operation. */
  virtual void* get_compute_stream() const { return DEFAULT_STREAM; }
  /** Return the run queue type this operation should use. */
  virtual RunType get_run_type() const { return RunType::bounded; }
  /** True if this is meant to block operations until completion. */
  virtual bool blocks() const { return false; }
  /** Return a name identifying the state (for debugging/info purposes). */
  virtual std::string get_name() const { return "AlState"; }
  /** Return a string description of the state (for debugging/info purposes). */
  virtual std::string get_desc() const { return ""; }
 private:
  AlRequest req;
#ifdef AL_DEBUG_HANG_CHECK
  bool hang_reported = false;
  double start_time = std::numeric_limits<double>::max();
#endif
  profiling::ProfileRange prof_range;
  /** Whether execution of this operation is paused on pipeline advancement. */
  bool paused_for_advance = false;
};

/**
 * Lock-free single-producer, single-consumer queue.
 * This is Lamport's classic SPSC queue, with memory order optimizations
 * (see Le, et al. "Correct and Efficient Bounded FIFO Queues").
 */
class SPSCQueue {
  friend class ProgressEngine;
 public:
  /**
   * Initialize the queue.
   * size_ must be a power of 2.
   */
  SPSCQueue(size_t size_) :
    front(0), back(0), size(size_) {
    data = new AlState*[size];
    std::fill_n(data, size, nullptr);
  }
  ~SPSCQueue() {
    delete[] data;
  }
  /**
   * Add v to the queue.
   * This will throw an exception if the queue is full.
   * @todo We can elide the capacity check to save an atomic load.
   * @todo If we elide the capacity check, this can be reformulated to use
   * just a single fetch-and-add.
   */
  void push(AlState* v) {
    size_t b = back.load(std::memory_order_relaxed);
    size_t f = front.load(std::memory_order_acquire);
    size_t bmod = (b+1) & (size-1);
    if (bmod == f) {
      throw_al_exception("Queue full");
    }
    data[b] = v;
    back.store(bmod, std::memory_order_release);
  }
  /**
   * Remove an element from the queue.
   * If the queue is empty, this returns nullptr.
   */
  AlState* pop() {
    size_t f = front.load(std::memory_order_relaxed);
    size_t b = back.load(std::memory_order_acquire);
    if (b == f) {
      return nullptr;
    }
    AlState* v = data[f];
    front.store((f+1) & (size-1), std::memory_order_release);
    return v;
  }
  /**
   * Discard the element at the front of the queue.
   * This always advances the front, even if nothing is present. The user must
   * ensure that something was actually there (via peek).
   */
  void pop_always() {
    size_t f = front.load(std::memory_order_relaxed);
    front.store((f+1) & (size-1), std::memory_order_release);
  }
  /**
   * Return the element at the front of the queue without removing it.
   * If the queue is empty, this returns nullptr.
   */
  AlState* peek() {
    size_t f = front.load(std::memory_order_relaxed);
    size_t b = back.load(std::memory_order_acquire);
    if (b == f) {
      return nullptr;
    }
    return data[f];
  }
 private:
  /** Index for the current front of the queue. */
  std::atomic<size_t> front;
  /** Index for the current back of the queue. */
  std::atomic<size_t> back;
  /** Number of elements the queue can store. */
  const size_t size;
  /** Buffer for data in the queue. */
  AlState** data;
};

/** Input request queue. */
struct InputQueue {
  InputQueue() : q(1<<13) {}
  /** Input queue. */
  SPSCQueue q;
  /** Whether a blocking operation is being executed. */
  bool blocked = false;
  /** Associated compute stream. */
  void* compute_stream = DEFAULT_STREAM;
};

/**
 * An fixed-length ordered array that allows arbitrary elements to be removed.
 * This is meant to be used for small N.
 */
template <size_t N>
struct OrderedArray {
  OrderedArray() {
    std::fill_n(l, N, nullptr);
  }
  ~OrderedArray() {}
  /** Whether the array is currently full. */
  bool full() const { return cur_size == N; }
  /**
   * Add an element to the end of the array.
   * This assumes the array is not full.
   */
  void push(AlState* v) {
    l[cur_size] = v;
    ++cur_size;
  }
  /**
   * Delete an entry.
   * This assumes idx exists.
   */
  void del(size_t idx) {
    // Just copy the elements over.
    for (size_t i = idx; i < cur_size - 1; ++i) {
      l[i] = l[i+1];
    }
    --cur_size;
  }
  /**
   * Fill "holes" created by marking elements null.
   */
  void compact() {
    size_t free_slot = 0;
    for (size_t i = 0; i < cur_size; ++i) {
      if (l[i] != nullptr) {
        // l[i] has something, see if we can move it into the free slot.
        if (free_slot < i) {
          l[free_slot] = l[i];
        }
        // Advance the free slot. If we copied something over, everything after
        // should be shifted too.
        ++free_slot;
      }
    }
    cur_size = free_slot;
  }
  /** Underlying data store. */
  AlState* l[N];
  /** Number of elements currently present. */
  size_t cur_size = 0;
};

/**
 * Encapsulates the asynchronous progress engine.
 * Note this is intended to be used from only one thread (in addition to the
 * progress thread) and is not optimized (or tested) for other cases.
 */
class ProgressEngine {
 public:
  ProgressEngine();
  ProgressEngine(const ProgressEngine&) = delete;
  ProgressEngine& operator=(const ProgressEngine) = delete;
  ~ProgressEngine();
  /** Start the progress engine. */
  void run();
  /** Stop the progress engine. */
  void stop();
  /** Enqueue state for asynchronous execution. */
  void enqueue(AlState* state);
  /**
   * Check whether a request has completed.
   * If the request is completed, it is removed.
   * This does not block and may spuriously return false.
   */
  bool is_complete(AlRequest& req);
  /**
   * Wait until a request has completed, then remove it.
   * This will block the calling thread.
   */
  void wait_for_completion(AlRequest& req);

  /**
   * Best effort to dump progress engine state for debugging.
   * State is written to ss.
   */
  std::ostream& dump_state(std::ostream& ss);
 private:
  /** The actual thread of execution. */
  std::thread thread;
  /** Atomic flag indicating the progress engine should stop; true to stop. */
  std::atomic<bool> stop_flag;
  /** For startup_cv. */
  std::mutex startup_mutex;
  /** Used to signal to the main thread that the progress engine has started. */
  std::condition_variable startup_cv;
  /** Atomic flag indicating that the progress engine has completed startup. */
  std::atomic<bool> started_flag;
  /**
   * Per-stream request queues.
   * Each queue contains requests that have been enqueued to the progress
   * engine but that it has not yet begun to process.
   * Queues can be "added" to this by incrementing cur_streams. The calling
   * thread sets up the queue, then increments cur_streams. (Note that this is
   * only safe with one user thread.)
   */
  InputQueue request_queues[AL_PE_NUM_STREAMS];
  /** Current number of streams. */
  std::atomic<size_t> num_input_streams;
  /**
   * Compute streams that currently have InputQueues.
   * Only the user thread accesses this.
   */
  std::unordered_map<void*, InputQueue*> stream_to_queue;
  /**
   * Per-stream pipelined run queues.
   * This should be accessed only by the progress engine.
   * Using a vector for compactness and to avoid repeated memory allocations.
   * @todo May extend OrderedArray / make a new class to handle this.
   */
  std::unordered_map<void*, std::array<std::vector<AlState*>, AL_PE_NUM_PIPELINE_STAGES>> run_queues;
  /** Number of currently-active bounded-length operations. */
  size_t num_bounded = 0;
  /**
   * Map requests that are currently blocking to the associated input queue.
   * This should be accessed only by the progress engine.
   */
  std::unordered_map<AlState*, size_t> blocking_reqs;
  /** World communicator. */
  mpi::MPICommunicator* world_comm;
#ifdef AL_HAS_CUDA
  /** Used to pass the original CUDA device to the progress engine thread. */
  std::atomic<int> cur_device;
#endif
  /**
   * Bind the progress engine to a core.
   * This binds to the last core in the NUMA node the process is in.
   * If there are multiple ranks per NUMA node, they get the last-1, etc. core.
   */
  void bind();
  /** This is the main progress engine loop. */
  void engine();
};

/** Return a pointer to the Aluminum progress engine. */
ProgressEngine* get_progress_engine();

}  // namespace internal
}  // namespace Al
