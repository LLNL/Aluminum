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

/**
 * Request handle for non-blocking operations.
 * The atomic flag is used to check for completion.
 */
using AlRequest = std::shared_ptr<std::atomic<bool>>;
/** Return a free request for use. */
AlRequest get_free_request();
/** Special marker for null requests. */
static constexpr std::nullptr_t NULL_REQUEST = nullptr;

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
  AlRequest& get_req() { return req; }
  /** True if this is meant ot be waited on by the user. */
  virtual bool needs_completion() const { return true; }
 private:
  AlRequest req;
};

/**
 * Lock-free single-producer, single-consumer queue.
 * This is Lamport's classic SPSC queue, with memory order optimizations
 * (see Le, et al. "Correct and Efficient Bounded FIFO Queues").
 */
class SPSCQueue {
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
   */
  SPSCQueue enqueued_reqs;
  /**
   * Requests the progress engine is currently processing.
   * This should be accessed only by the progress engine (it is not protected).
   */
  AlState* in_progress_reqs[AL_PE_NUM_CONCURRENT_OPS];
  /** Number of requests currently being processed. */
  size_t num_in_progress_reqs = 0;
  /** World communicator. */
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
