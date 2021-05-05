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

#include <type_traits>
#include <array>

namespace Al {
namespace internal {

/** True if x is a power of 2 (0 is not a power of 2). */
template <typename T>
constexpr bool is_pow2(T x) {
  static_assert(std::is_integral<T>::value, "T must be integral");
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  return x && !(x & (x - 1));
}

/** The next highest power of 2, or x if x is a power of 2. */
template <typename T>
constexpr typename std::enable_if<sizeof(T) == 4, T>::type
next_highest_pow2(T x) {
  static_assert(std::is_integral<T>::value, "T must be integral");
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

template <typename T>
constexpr typename std::enable_if<sizeof(T) == 8, T>::type
next_highest_pow2(T x) {
  static_assert(std::is_integral<T>::value, "T must be integral");
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

/** The floor of log base 2 of x. */
template <typename T>
constexpr T floor_log2(T x) {
  static_assert(std::is_integral<T>::value, "T must be integral");
  static_assert(std::is_unsigned<T>::value, "T must be unsigned");
  T b = 0;
  while (x >>= 1) {
    ++b;
  }
  return b;
}

/** The ceiling of log base 2 of x. */
template <typename T>
constexpr T ceil_log2(T x) {
  T b = floor_log2(x);
  return is_pow2(x) ? b : b + 1;
}

/** Return the number of powers of 2 between Start and End, inclusive. */
template <typename T, T Start, T End>
constexpr T num_pow2s_between() {
  static_assert(End >= Start, "Must have End >= Start");
  return floor_log2(End) - floor_log2(Start) + (is_pow2(Start) ? 1 : 0);
}

/** Return an array of all powers of 2 between Start and End, inclusive. */
template <typename T, T Start, T End>
constexpr std::array<T, num_pow2s_between<T, Start, End>()> pow2_ar() {
  static_assert(Start != 0 && End != 0, "Start and End must be non-zero");
  constexpr size_t N = num_pow2s_between<T, Start, End>();
  std::array<T, N> a{};
  for (size_t i = 0, v = next_highest_pow2(Start); i < N; ++i, v <<= 1) {
    a[i] = v;
  }
  return a;
}

}  // namespace internal
}  // namespace Al
