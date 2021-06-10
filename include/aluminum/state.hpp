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

#include <memory>
#include <atomic>

#include "aluminum/profiling.hpp"

namespace Al {
namespace internal {

// TODO: Get rid of AlRequest, since it's only used by the MPI backend.

/**
 * Request handle for non-blocking operations.
 * The atomic flag is used to check for completion.
 */
using AlRequest = std::shared_ptr<std::atomic<bool>>;
/** Return a free request for use. */
inline AlRequest get_free_request() {
  return std::make_shared<std::atomic<bool>>(false);
}
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
  virtual ~AlState() { profiling::prof_end(prof_range); }
  /**
   * Perform initial setup of the algorithm.
   * This is called by the progress engine when the operation begins execution.
   */
  virtual void start() { prof_range = profiling::prof_start(get_name()); }
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

}  // namespace internal
}  // namespace Al
