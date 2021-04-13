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

#include <string>
#include <sstream>

#include <Al_config.hpp>
#include "aluminum/utils/utils.hpp"

namespace Al {
namespace internal {
// Forward declaration.
class AlState;
namespace trace {

#ifdef AL_TRACE
// Need to be able to print vectors.
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  if (!v.empty()) {
    for (size_t i = 0; i < v.size() - 1; ++i) {
      os << v[i] << ", ";
    }
    os << v[v.size() - 1];
  }
  os << "]";
  return os;
}
#endif

/**
 * Save entry to the trace log.
 * progress is whether this comes from the progress engine, which is recorded
 * separately.
 */
void save_trace_entry(std::string entry, bool progress = false);

/** Record an operation to the trace log. */
template <typename Backend, typename T, typename... Args>
#ifdef AL_TRACE
void record_op(std::string op, typename Backend::comm_type& comm, Args... args) {
  std::stringstream ss;
  ss << get_time() << ": "
     << Backend::Name() << " "
     << typeid(T).name() << " "
     << op << " "
     << comm.rank() << " " << comm.size() << " ";
  // See:
  // https://stackoverflow.com/questions/27375089/what-is-the-easiest-way-to-print-a-variadic-parameter-pack-using-stdostream
  using expander = int[];
  (void) expander{0, (void(ss << " " << std::forward<Args>(args)), 0)...};
  save_trace_entry(ss.str(), false);
}
#else  // AL_TRACE
void record_op(std::string, typename Backend::comm_type&, Args...) {
}
#endif  // AL_TRACE

/** Record a progress engine operation start to the trace log. */
void record_pe_start(const AlState& state);
/** Record a progress engine operation completion to the trace log. */
void record_pe_done(const AlState& state);

/** Write trace logs to os. */
std::ostream& write_trace_log(std::ostream& os);
/** Write trace logs to hostname.pid.trace.txt. */
void write_trace_to_file();

}  // namespace trace
}  // namespace internal
}  // namespace Al
