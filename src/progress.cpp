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

#include <hwloc.h>
#include <cstdlib>
#include "Al.hpp"
#include "progress.hpp"

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

namespace Al {
namespace internal {

AlRequest get_free_request() {
  return std::make_shared<std::atomic<bool>>(false);
}

ProgressEngine::ProgressEngine() : enqueued_reqs(1<<16) {
  stop_flag = false;
  started_flag = false;
  world_comm = new MPICommunicator(MPI_COMM_WORLD);
}

ProgressEngine::~ProgressEngine() {
  delete world_comm;
}

void ProgressEngine::run() {
#ifdef AL_HAS_CUDA
  // Capture the current CUDA device for the progress engine.
  int device;
  AL_CHECK_CUDA(cudaGetDevice(&device));
  cur_device = device;
#endif
  thread = std::thread(&ProgressEngine::engine, this);
  // Wait for the progress engine to start.
  std::unique_lock<std::mutex> lock(startup_mutex);
  startup_cv.wait(lock, [this] {return started_flag.load() == true;});
}

void ProgressEngine::stop() {
  if (stop_flag.load()) {
    throw_al_exception("Stop called twice on progress engine");
  }
  stop_flag.store(true, std::memory_order_release);
  thread.join();
}

void ProgressEngine::enqueue(AlState* state) {
  enqueued_reqs.push(state);
}

bool ProgressEngine::is_complete(AlRequest& req) {
  if (req == NULL_REQUEST) {
    return true;
  }
  if (req->load(std::memory_order_acquire)) {
    req = NULL_REQUEST;
    return true;
  }
  return false;
}

void ProgressEngine::wait_for_completion(AlRequest& req) {
  if (req == NULL_REQUEST) {
    return;
  }
  // Spin until the request has completed.
  while (!req->load(std::memory_order_acquire)) {}
  req = NULL_REQUEST;
}

void ProgressEngine::bind() {
  // Determine topology information.
  hwloc_topology_t topo;
  hwloc_topology_init(&topo);
  hwloc_topology_load(topo);
  // Determine how many NUMA nodes there are.
  int num_numa_nodes = hwloc_get_nbobjs_by_type(topo, HWLOC_OBJ_NUMANODE);
  if (num_numa_nodes == -1) {
    throw_al_exception("Cannot determine number of NUMA nodes.");
  }
  // Determine the NUMA node we're currently on.
  hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
  hwloc_get_cpubind(topo, cpuset, 0);
  hwloc_nodeset_t nodeset = hwloc_bitmap_alloc();
  hwloc_cpuset_to_nodeset(topo, cpuset, nodeset);
  hwloc_bitmap_singlify(nodeset);
  hwloc_obj_t numa_node = hwloc_get_numanode_obj_by_os_index(
    topo, hwloc_bitmap_first(nodeset));
  if (numa_node == NULL) {
    throw_al_exception("Could not get NUMA node.");
  }
  int core_to_bind = -1;
  // Check if the core has been manually set.
  char* env = std::getenv("AL_PROGRESS_CORE");
  if (env) {
    // Note: This still binds within the current NUMA node.
    core_to_bind = std::atoi(env);
  } else {
    // Determine how many cores are in this NUMA node.
    int num_cores = hwloc_get_nbobjs_inside_cpuset_by_type(
      topo, numa_node->cpuset, HWLOC_OBJ_CORE);
    if (num_cores <= 0) {
      throw_al_exception("Could not determine number of cores.");
    }
    // Determine which core on this NUMA node to map us to.
    // Support specifying this in the environment too.
    int ranks_per_numa_node = -1;
    env = std::getenv("AL_PROGRESS_RANKS_PER_NUMA_NODE");
    if (env) {
      ranks_per_numa_node = std::atoi(env);
    } else {
      // Note: This doesn't handle the case where things aren't evenly divisible.
      ranks_per_numa_node = std::max(
        1, world_comm->local_size() / num_numa_nodes);
    }
    int numa_rank = world_comm->local_rank() % ranks_per_numa_node;
    if (numa_rank > num_cores) {
      throw_al_exception("Not enough cores to bind to.");
    }
    // Assume the NUMA node is partitioned among the ranks on it, and bind to
    // the last core in our chunk.
    core_to_bind = (numa_rank + 1)*(num_cores / ranks_per_numa_node) - 1;
  }
  hwloc_obj_t core = hwloc_get_obj_inside_cpuset_by_type(
    topo, numa_node->cpuset, HWLOC_OBJ_CORE, core_to_bind);
  if (core == NULL) {
    throw_al_exception("Could not get core.");
  }
  hwloc_cpuset_t coreset = hwloc_bitmap_dup(core->cpuset);
  hwloc_bitmap_singlify(coreset);
  if (hwloc_set_cpubind(topo, coreset, HWLOC_CPUBIND_THREAD) == -1) {
    throw_al_exception("Cannot bind progress engine");
  }
  hwloc_bitmap_free(cpuset);
  hwloc_bitmap_free(nodeset);
  hwloc_bitmap_free(coreset);
  hwloc_topology_destroy(topo);
}

void ProgressEngine::engine() {
#ifdef AL_HAS_CUDA
  // Set the current CUDA device for the thread.
  AL_CHECK_CUDA_NOSYNC(cudaSetDevice(cur_device.load()));
#endif
  bind();
  // Notify the main thread we're now running.
  {
    std::unique_lock<std::mutex> lock(startup_mutex);
    started_flag = true;
  }
  startup_cv.notify_one();
  while (!stop_flag.load(std::memory_order_acquire)) {
    // Check for newly-submitted requests, if we can take more.
    if (!in_progress_reqs.full()) {
      AlState* req = enqueued_reqs.pop();
      if (req != nullptr) {
        in_progress_reqs.push(req);
        req->start();
      }
    }
    // Process one step of each in-progress request.
    bool any_completed = false;
    for (size_t i = 0; i < in_progress_reqs.cur_size; ++i) {
      if (in_progress_reqs.l[i]->step()) {
        if (in_progress_reqs.l[i]->needs_completion()) {
          // Mark the request as completed.
          in_progress_reqs.l[i]->get_req()->store(true, std::memory_order_release);
        }
        delete in_progress_reqs.l[i];
        // Mark slot for compaction.
        in_progress_reqs.l[i] = nullptr;
        any_completed = true;
      }
    }
    if (any_completed) {
      in_progress_reqs.compact();
    }
  }
}

}  // namespace internal
}  // namespace Al
