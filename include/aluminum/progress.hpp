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

#include "aluminum/base.hpp"
#include "aluminum/tuning_params.hpp"
#include "aluminum/mpi/communicator.hpp"
#include "aluminum/state.hpp"
#include "aluminum/utils/spsc_queue.hpp"
#include "aluminum/utils/mpsc_queue.hpp"

namespace Al {
namespace internal {

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
  /** Input request queue. */
  struct InputQueue {
    InputQueue() : q(AL_PE_INPUT_QUEUE_SIZE) {}
    /** Input queue. */
#ifdef AL_THREAD_MULTIPLE
    MPSCQueue<AlState*> q;
#else
    SPSCQueue<AlState*> q;
#endif
    /** Whether a blocking operation is being executed. */
    bool blocked = false;
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
