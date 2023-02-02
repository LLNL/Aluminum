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

#include <exception>
#include <string>

/** HOST_NAME_MAX is a linux only define */
#ifndef HOST_NAME_MAX
# if defined(_POSIX_HOST_NAME_MAX)
#  define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
# elif defined(MAXHOSTNAMELEN)
#  define HOST_NAME_MAX MAXHOSTNAMELEN
# endif
#endif /* HOST_NAME_MAX */

/** Intentionally ignore results of [[nodiscard]] functions. */
#define AL_IGNORE_NODISCARD(fcall) static_cast<void>((fcall))

namespace Al {

/**
 * Base Aluminum exception class.
 */
class al_exception : public std::exception {
 public:
  al_exception(const std::string m, const std::string f, const int l) :
    msg(m), file(f), line(l) {
    err = file + ":" + std::to_string(line) + " - " + msg;
  }
  const char* what() const noexcept override {
    return err.c_str();
  }
private:
  /** Exception message. */
  const std::string msg;
  /** File exception occurred in. */
  const std::string file;
  /** Line exception occurred at. */
  const int line;
  /** Constructed error message. */
  std::string err;
};
#define throw_al_exception(s) throw Al::al_exception(s, __FILE__, __LINE__)

/** Predefined reduction operations. */
enum class ReductionOperator {
  sum, prod, min, max, lor, land, lxor, bor, band, bxor, avg
};

} // namespace Al
