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

#include <chrono>
#include <vector>

namespace Al {

/** Return time, in seconds (with decimal), since a fixed epoch. */
inline double get_time() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(
    steady_clock::now().time_since_epoch()).count();
}

/**
 * Compute an exclusive prefix sum.
 *
 * This is mostly meant to help with vector collectives.
 */
template <typename T>
inline std::vector<T> excl_prefix_sum(const std::vector<T>& v) {
  auto r = std::vector<T>(v.size(), T{0});
  for (size_t i = 1; i < v.size(); ++i) {
    r[i] = v[i-1] + r[i-1];
  }
  return r;
}

}  // namespace Al
