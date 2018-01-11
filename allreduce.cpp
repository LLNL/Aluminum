#include "allreduce.hpp"

namespace allreduces {

namespace {
// Whether the library has been initialized.
bool is_initialized = false;
// Progress engine.
internal::ProgressEngine* progress_engine = nullptr;
}

void Initialize(int& argc, char**& argv) {
  progress_engine = new internal::ProgressEngine();
  progress_engine->run();
  internal::mpi::init(argc, argv);
  is_initialized = true;
}

void Finalize() {
  internal::mpi::finalize();
  progress_engine->stop();
  delete progress_engine;
  progress_engine = nullptr;
  is_initialized = false;
}

bool Initialized() {
  return is_initialized;
}

bool Test(AllreduceRequest req) {
  internal::ProgressEngine* pe = internal::get_progress_engine();
  return pe->is_complete(req);
}

void Wait(AllreduceRequest req) {
  internal::ProgressEngine* pe = internal::get_progress_engine();
  pe->wait_for_completion(req);
}

namespace internal {

AllreduceRequest get_free_request() {
  static AllreduceRequest cur_req = 1;
  return cur_req++;
}

void ProgressEngine::run() {
  thread = std::thread(&ProgressEngine::engine, this);
}

void ProgressEngine::stop() {
  if (stop_flag.load()) {
    throw allreduce_exception("Stop called twice on progress engine");
  }
  stop_flag = true;
#if ALLREDUCE_PE_SLEEPS
  enqueue_cv.notify_one();  // Wake up the engine if needed.
#endif
  thread.join();
}

void ProgressEngine::enqueue(AllreduceState* state) {
  enqueue_mutex.lock();
  enqueued_reqs.push(state);
  enqueue_mutex.unlock();
#if ALLREDUCE_PE_SLEEPS
  enqueue_cv.notify_one();  // Wake up the engine if needed.
#endif
}

bool ProgressEngine::is_complete(AllreduceRequest& req) {
  if (req == NULL_REQUEST) {
    return true;
  }
  if (completed_mutex.try_lock()) {
    auto i = completed_reqs.find(req);
    if (i != completed_reqs.end()) {
      AllreduceState* state = i->second;
      completed_reqs.erase(i);
      completed_mutex.unlock();
      delete state;
      req = NULL_REQUEST;
      return true;
    } else {
      completed_mutex.unlock();
    }
  }
  return false;
}

void ProgressEngine::wait_for_completion(AllreduceRequest& req) {
  if (req == NULL_REQUEST) {
    return;
  }
  // First check if the request is already complete.
  std::unique_lock<std::mutex> lock(completed_mutex);
  auto i = completed_reqs.find(req);
  if (i != completed_reqs.end()) {
    delete i->second;
    completed_reqs.erase(i);
    req = NULL_REQUEST;
    return;
  }
  // Request not complete, wait on the cv until something completes and see
  // if it is req.
  while (true) {
    completion_cv.wait(lock);
    i = completed_reqs.find(req);
    if (i != completed_reqs.end()) {
      delete i->second;
      completed_reqs.erase(i);
      req = NULL_REQUEST;
      return;
    }
  }
}

void ProgressEngine::bind() {
  // TODO
}

void ProgressEngine::engine() {
  bind();
  while (!stop_flag.load()) {
    // Check for newly-submitted requests. Grab one per iteration.
    // Don't block if someone else has the lock.
    if (in_progress_reqs.empty() && enqueue_mutex.try_lock()) {
#if ALLREDUCE_PE_SLEEPS
      if (in_progress_reqs.empty() && enqueued_reqs.empty()) {
        // No work to do, so instead of spinning, wait.
        std::unique_lock<std::mutex> lock(enqueue_mutex, std::adopt_lock);
        enqueue_cv.wait(lock);
      }
#endif
      if (!enqueued_reqs.empty()) {
        in_progress_reqs.push_back(enqueued_reqs.front());
        enqueued_reqs.pop();
      }
      enqueue_mutex.unlock();
    }
    std::vector<AllreduceState*> completed;
    // Process one step of each in-progress request.
    for (auto i = in_progress_reqs.begin(); i != in_progress_reqs.end();) {
      AllreduceState* state = *i;
      if (state->step()) {
        // Request completed, but don't try to block here.
        completed.push_back(state);
        i = in_progress_reqs.erase(i);
      } else {
        ++i;
      }
    }
    // Shift over completed requests.
    if (!completed.empty()) {
      completed_mutex.lock();
      for (AllreduceState* state : completed) {
        completed_reqs[state->get_req()] = state;
      }
      completed_mutex.unlock();
      completion_cv.notify_all();
    }
  }
}

ProgressEngine* get_progress_engine() {
  return progress_engine;
}

}  // namespace internal
}  // namespace allreduces
