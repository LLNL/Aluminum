#include <hwloc.h>
#include "allreduce.hpp"

// For ancient versions of hwloc.
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
// Ported from more recent hwloc versions.
hwloc_obj_t hwloc_get_numanode_obj_by_os_index(hwloc_topology_t topology, unsigned os_index) {
  hwloc_obj_t obj = NULL;
  while ((obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, obj)) != NULL) {
    if (obj->os_index == os_index) {
      return obj;
    }
  }
  return NULL;
}
#endif

namespace allreduces {

namespace {
// Whether the library has been initialized.
bool is_initialized = false;
// Progress engine.
internal::ProgressEngine* progress_engine = nullptr;
}

void Initialize(int& argc, char**& argv) {
  internal::mpi::init(argc, argv);
  progress_engine = new internal::ProgressEngine();
  progress_engine->run();
  is_initialized = true;
}

void Finalize() {
  progress_engine->stop();
  delete progress_engine;
  progress_engine = nullptr;
  is_initialized = false;
  internal::mpi::finalize();
}

bool Initialized() {
  return is_initialized;
}

namespace internal {

AllreduceRequest get_free_request() {
  static AllreduceRequest cur_req = 1;
  return cur_req++;
}

ProgressEngine::ProgressEngine() {
  stop_flag = false;
  started_flag = false;
  world_comm = new MPICommunicator(MPI_COMM_WORLD);
}

ProgressEngine::~ProgressEngine() {
  delete world_comm;
}

void ProgressEngine::run() {
  thread = std::thread(&ProgressEngine::engine, this);
  // Wait for the progress engine to start.
  std::unique_lock<std::mutex> lock(startup_mutex);
  startup_cv.wait(lock, [this] {return started_flag.load() == true;});
}

void ProgressEngine::stop() {
  if (stop_flag.load()) {
    throw_allreduce_exception("Stop called twice on progress engine");
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
    AllreduceState* state = i->second;
    completed_reqs.erase(i);
    lock.unlock();
    delete state;
    req = NULL_REQUEST;
    return;
  }
  // Request not complete, wait on the cv until something completes and see
  // if it is req.
  while (true) {
    completion_cv.wait(lock);
    i = completed_reqs.find(req);
    if (i != completed_reqs.end()) {
      AllreduceState* state = i->second;
      completed_reqs.erase(i);
      lock.unlock();
      delete state;
      req = NULL_REQUEST;
      return;
    }
  }
}

void ProgressEngine::bind() {
  // Determine topology information.
  hwloc_topology_t topo;
  hwloc_topology_init(&topo);
  hwloc_topology_load(topo);
  // Determine how many NUMA nodes there are.
  int num_numa_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
  if (num_numa_nodes == -1) {
    throw_allreduce_exception("Cannot determine number of NUMA nodes.");
  }
  // Determine the NUMA node we're currently on.
  hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
  hwloc_get_cpubind(topo, cpuset, 0);
  hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
  hwloc_cpuset_to_nodeset(topo, cpuset, nodeset);
  hwloc_bitmap_singlify(nodeset);
  hwloc_obj_t numa_node = hwloc_get_numanode_obj_by_os_index(
    topo, hwloc_bitmap_first(nodeset));
  // Determine how many cores are in this NUMA node.
  int num_cores = hwloc_get_nbobjs_inside_cpuset_by_type(
    topo, numa_node->cpuset, HWLOC_OBJ_CORE);
  // Determine which core on this NUMA node to map us to.
  // Note: This doesn't handle the case where things aren't evenly divisible.
  int ranks_per_numa_node = std::max(
    1, world_comm->local_size() / num_numa_nodes);
  int numa_rank = world_comm->local_rank() % ranks_per_numa_node;
  // Pin to the last - numa_rank core.
  hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(
    topo, numa_node->cpuset, HWLOC_OBJ_CORE, num_cores - 1 - numa_rank);
  hwloc_cpuset_t coreset = hwloc_bitmap_dup(core->cpuset);
  hwloc_bitmap_singlify(coreset);
  if (hwloc_set_cpubind(topo, coreset, HWLOC_CPUBIND_THREAD) == -1) {
    throw_allreduce_exception("Cannot bind progress engine");
  }
  hwloc_bitmap_free(cpuset);
  hwloc_bitmap_free(nodeset);
  hwloc_bitmap_free(coreset);
  hwloc_topology_destroy(topo);
}

void ProgressEngine::engine() {
  bind();
  started_flag = true;
  startup_cv.notify_one();
  while (!stop_flag.load()) {
    // Check for newly-submitted requests, if we can take more.
    if (ALLREDUCE_PE_NUM_CONCURRENT_ALLREDUCES == 0 ||
         in_progress_reqs.size() < ALLREDUCE_PE_NUM_CONCURRENT_ALLREDUCES) {
      // Don't block if someone else has the lock.
      std::unique_lock<std::mutex> lock(enqueue_mutex, std::try_to_lock);
      if (lock) {
#if ALLREDUCE_PE_SLEEPS
        // If there's no work, we sleep.
        if (in_progress_reqs.empty() && enqueued_reqs.empty()) {
          enqueue_cv.wait(
            lock, [this] {
              return (!enqueued_reqs.empty() &&
                      (ALLREDUCE_PE_NUM_CONCURRENT_ALLREDUCES == 0 ||
                       in_progress_reqs.size() < ALLREDUCE_PE_NUM_CONCURRENT_ALLREDUCES)) ||
                stop_flag.load();
            });
        }
#endif
        while (!enqueued_reqs.empty() &&
               (ALLREDUCE_PE_NUM_CONCURRENT_ALLREDUCES == 0 ||
                in_progress_reqs.size() < ALLREDUCE_PE_NUM_CONCURRENT_ALLREDUCES)) {
          in_progress_reqs.push_back(enqueued_reqs.front());
          enqueued_reqs.pop();
        }
      }
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
