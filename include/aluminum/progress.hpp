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

/**
 * @file
 * Asynchronous progress engine.
 */

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <ostream>
#include <thread>
#include <unordered_map>
#include <vector>

#include <Al_config.hpp>
#include "aluminum/tuning_params.hpp"
#include "aluminum/state.hpp"
#ifdef AL_THREAD_MULTIPLE
#include "aluminum/utils/mpsc_queue.hpp"
#else
#include "aluminum/utils/spsc_queue.hpp"
#endif

namespace Al {
namespace internal {

// Forward declaration:
namespace mpi {
class MPICommunicator;
}

/**
 * Encapsulates the asynchronous progress engine.
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
   * Best effort to dump progress engine state for debugging.
   * State is written to ss.
   */
  std::ostream& dump_state(std::ostream& ss);
 private:
  /** Input request queue. */
  struct InputQueue {
    InputQueue() : q(AL_PE_INPUT_QUEUE_SIZE) {}
    /** Input queue. */
#ifdef AL_THREAD_MULTIPLE
    MPSCQueue<AlState*> q;
#else
    SPSCQueue<AlState*> q;
#endif
    /** Associated compute stream. */
    void* compute_stream = DEFAULT_STREAM;
  };

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
#ifdef AL_PE_START_ON_DEMAND
  /** Atomic flag indicating that a thread is starting the progess engine. */
  std::atomic<bool> doing_start_flag;
#endif
  /**
   * Per-stream request queues.
   *
   * Each queue contains requests that have been enqueued to the progress
   * engine but that it has not yet begun to process.
   * There should be only one queue per stream.
   * Queues can be "added" to this by incrementing cur_streams. The calling
   * thread sets up the queue, then increments cur_streams. The
   * add_queue_mutex mutex is used to synchronize this operation. It
   * should be rare.
   */
  InputQueue request_queues[AL_PE_NUM_STREAMS];
#ifdef AL_THREAD_MULTIPLE
  /** Synchronize adding a new queue. */
  std::mutex add_queue_mutex;
#endif
  /** Current number of streams. */
  std::atomic<size_t> num_input_streams;
#ifdef AL_PE_STREAM_QUEUE_CACHE
  /** Per-thread mapping from streams to queues (cached to avoid lookup). */
#ifdef AL_THREAD_MULTIPLE
  static thread_local std::unordered_map<void*, InputQueue*> stream_to_queue;
#else
  static std::unordered_map<void*, InputQueue*> stream_to_queue;
#endif
#endif
  /**
   * Per-stream pipelined run queues.
   * This should be accessed only by the progress engine.
   * Using a vector for compactness and to avoid repeated memory allocations.
   */
  std::unordered_map<void*, std::array<std::vector<AlState*>, AL_PE_NUM_PIPELINE_STAGES>> run_queues;
  /** Number of currently-active bounded-length operations. */
  size_t num_bounded = 0;

#ifdef AL_USE_HWLOC
  /** Core to bind the progress engine to. */
  int core_to_bind = -1;
#endif

#ifdef AL_HAS_CUDA
  /** Used to pass the original CUDA device to the progress engine thread. */
  std::atomic<int> cur_device;
#endif

#ifdef AL_USE_HWLOC
  /** Initialize progress engine binding (must be called before bind). */
  void bind_init();
  /**
   * Bind the progress engine to a core.
   * This binds to the last core in the NUMA node the process is in.
   * If there are multiple ranks per NUMA node, they get the last-1, etc. core.
   */
  void bind();
#endif

  /** This is the main progress engine loop. */
  void engine();
};

/** Return a pointer to the Aluminum progress engine. */
ProgressEngine* get_progress_engine();

}  // namespace internal
}  // namespace Al
