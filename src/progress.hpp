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

namespace Al {

// Forward declaration.
class Communicator;

namespace internal {

/** Request handle for non-blocking operations.. */
using AlRequest = int;  // TODO: This is a placeholder.

/** Return a free request for use. */
AlRequest get_free_request();
/** Special marker for null requests. */
static const AlRequest NULL_REQUEST = 0;

/**
 * Represents the state and algorithm for an asynchronous operation.
 * A non-blocking operation should create one of these and enqueue it for
 * execution by the progress thread. Specific implementations can override
 * as needed.
 * An algorithm should be broken up into steps which execute some small,
 * discrete operation. Steps from different operations may be interleaved.
 * Note that the memory pool is not thread-safe, so memory from it should be
 * pre-allocated before enqueueing.
 */
class AlState {
 public:
  /** Create with an associated request. */
  AlState(AlRequest req_) : req(req_) {}
  virtual ~AlState() {}
  /**
   * Start one step of the algorithm.
   * Return true if the operation has completed, false if it has more steps
   * remaining.
   */
  virtual bool step() = 0;
  /** Return the associated request. */
  AlRequest get_req() const { return req; }
  /** True if this is meant ot be waited on by the user. */
  virtual bool needs_completion() const { return true; }
 private:
  AlRequest req;
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
   * The list of requests that have been enqueued to the progress engine, but
   * that it has not yet begun to process. Threads may add states for
   * asynchronous execution, and the progress engine will dequeue them when it
   * begins to run them.
   * This is protected by the enqueue_mutex.
   */
  std::queue<AlState*> enqueued_reqs;
  /** Protects enqueued_reqs. */
  std::mutex enqueue_mutex;
#if AL_PE_SLEEPS
  /**
   * The progress engine will sleep on this when there is otherwise no work to
   * do.
   */
  std::condition_variable enqueue_cv;
#endif
  /**
   * Requests the progress engine is currently processing.
   * This should be accessed only by the progress engine (it is not protected).
   */
  std::list<AlState*> in_progress_reqs;
  /**
   * Requests that have been completed.
   * The request is added by the progress engine once it has been completed.
   * States should be deallocated by whatever removes them from this.
   * This is protected by the completed_mutex.
   */
  std::unordered_map<AlRequest, AlState*> completed_reqs;
  /** Protects completed_reqs. */
  std::mutex completed_mutex;
  /** Used to notify any thread waiting for completion. */
  std::condition_variable completion_cv;
  /**
   * World communicator.
   */
  Communicator* world_comm;
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
