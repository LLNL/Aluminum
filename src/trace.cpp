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

#include <unistd.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <fstream>
#include "Al.hpp"

namespace Al {
namespace internal {
namespace trace {

namespace {
std::vector<std::string> trace_log;
std::vector<std::string> pe_trace_log;
}

void save_trace_entry(std::string entry, bool progress) {
  if (progress) {
    pe_trace_log.push_back(entry);
  } else {
    trace_log.push_back(entry);
  }
}

void record_pe_start(const AlState& state) {
#ifdef AL_TRACE
  std::stringstream ss;
  ss << get_time() << ": PE START "
     << state.get_name() << " "
     << state.get_desc();
  save_trace_entry(ss.str(), true);
#else
  (void) state;
#endif
}

void record_pe_done(const AlState& state) {
#ifdef AL_TRACE
  std::stringstream ss;
  ss << get_time() << ": PE DONE "
     << state.get_name() << " "
     << state.get_desc();
  save_trace_entry(ss.str(), true);
#else
  (void) state;
#endif
}

std::ostream& write_trace_log(std::ostream& os) {
#ifdef AL_TRACE
  os << "Trace:\n";
  for (const auto& entry : trace_log) os << entry << "\n";
  os << "Progress engine trace:\n";
  for (const auto& entry : pe_trace_log) os << entry << "\n";
  return os;
#else
  return os;
#endif
}

void write_trace_to_file() {
#ifdef AL_TRACE
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  pid_t pid = getpid();
  std::string filename = std::string(hostname) + "." + std::to_string(pid)
    + ".trace.txt";
  std::ofstream trace_file(filename);
  write_trace_log(trace_file);
#endif
}

}  // namespace trace
}  // namespace internal
}  // namespace Al
