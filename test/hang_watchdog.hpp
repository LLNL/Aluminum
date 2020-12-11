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

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstddef>


/**
 * Prints a warning unless it's told some event has completed within a set time.
 */
class HangWatchdog {
public:
  /** Hang with a timeout in seconds. */
  HangWatchdog(size_t timeout_ = 60, bool do_abort_ = true) :
    timeout(timeout_), do_abort(do_abort_) {
    watchdog = std::thread(&HangWatchdog::run, this);
  }

  ~HangWatchdog() {
    {
      std::unique_lock<std::mutex> lock(watchdog_mutex);
      stop_flag = true;
    }
    cv.notify_one();
    watchdog.join();
  }

  /** Start waiting for some event. */
  void start(std::string desc) {
    {
      std::unique_lock<std::mutex> lock(watchdog_mutex);
      if (event_started) {
        std::cerr << "Cannot start while event is running" << std::endl;
      }
      event_desc = desc;
      event_started = true;
    }
    // Notify watchdog to start watching.
    cv.notify_one();
    // Wait until the watchdog starts watching.
    {
      std::unique_lock<std::mutex> lock(watchdog_mutex);
      cv.wait(lock, [&] { return watching_hang; });
    }
  }

  /** Indicate the event has completed. */
  void finish() {
    {
      std::unique_lock<std::mutex> lock(watchdog_mutex);
      event_finished = true;
    }
    cv.notify_one();
    // Wait for the watchdog to finish.
    {
      std::unique_lock<std::mutex> lock(watchdog_mutex);
      cv.wait(lock, [&] { return !watching_hang; });
      event_finished = false;
    }
  }

private:
  size_t timeout;
  bool do_abort;
  std::thread watchdog;
  std::mutex watchdog_mutex;
  std::condition_variable cv;
  bool stop_flag = false;
  bool event_started = false;
  bool watching_hang = false;
  bool event_finished = false;
  std::string event_desc;

  void run() {
    while (true) {
      // Wait until something starts and acknowledge.
      {
        std::unique_lock<std::mutex> lock(watchdog_mutex);
        cv.wait(lock, [&] { return stop_flag || event_started; });
        if (stop_flag) {
          return;
        }
        watching_hang = true;
      }
      cv.notify_one();
      // Wait until notified or timeout.
      {
        std::unique_lock<std::mutex> lock(watchdog_mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(timeout),
                         [&] { return event_finished; })) {
          std::cerr << "Aborting after hang in " << event_desc << std::endl;
          if (do_abort) {
            std::abort();
          }
        }
        event_started = false;
        watching_hang = false;
      }
      // Ack we've seen the completion.
      cv.notify_one();
    }
  }
};
