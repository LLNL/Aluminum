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

#include <mpi.h>

namespace Al {
namespace internal {
namespace mpi {

/** Used to map types to the associated MPI datatype. */
template <typename T>
inline MPI_Datatype TypeMap();
template <> inline MPI_Datatype TypeMap<char>() { return MPI_CHAR; }
template <> inline MPI_Datatype TypeMap<signed char>() { return MPI_SIGNED_CHAR; }
template <> inline MPI_Datatype TypeMap<unsigned char>() { return MPI_UNSIGNED_CHAR; }
template <> inline MPI_Datatype TypeMap<short>() { return MPI_SHORT; }
template <> inline MPI_Datatype TypeMap<unsigned short>() { return MPI_UNSIGNED_SHORT; }
template <> inline MPI_Datatype TypeMap<int>() { return MPI_INT; }
template <> inline MPI_Datatype TypeMap<unsigned int>() { return MPI_UNSIGNED; }
template <> inline MPI_Datatype TypeMap<long int>() { return MPI_LONG; }
template <> inline MPI_Datatype TypeMap<unsigned long int>() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype TypeMap<long long int>() { return MPI_LONG_LONG_INT; }
template <> inline MPI_Datatype TypeMap<unsigned long long int>() { return MPI_UNSIGNED_LONG_LONG; }
template <> inline MPI_Datatype TypeMap<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype TypeMap<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype TypeMap<long double>() { return MPI_LONG_DOUBLE; }

/** Return either sendbuf or MPI_IN_PLACE. */
template <typename T>
void* buf_or_inplace(T* buf) {
  return buf == IN_PLACE<T>() ? MPI_IN_PLACE : buf;
}
/** Return either sendbuf or MPI_IN_PLACE. */
template <typename T>
const void* buf_or_inplace(const T* buf) {
  return buf == IN_PLACE<T>() ? MPI_IN_PLACE : buf;
}

/**
 * Convert a vector of size_ts to a vector of ints.
 *
 * This is used since MPI requires arrays of ints, which are a different size
 * than size_t.
 */
inline std::vector<int> intify_size_t_vector(const std::vector<size_t> v) {
  return std::vector<int>(v.begin(), v.end());
}

/** True if count elements can be sent by MPI. */
inline bool check_count_fits_mpi(size_t count) {
  return count <= static_cast<size_t>(std::numeric_limits<int>::max());
}
/** Throw an exception if count elements cannot be sent by MPI. */
inline void assert_count_fits_mpi(size_t count) {
  if (!check_count_fits_mpi(count)) {
    throw_al_exception("Message count too large for MPI");
  }
}

/** Basic sum reduction. */
template <typename T>
void sum_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] += src[i];
  }
}
/** Basic prod reduction. */
template <typename T>
void prod_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] *= src[i];
  }
}
/** Basic min reduction. */
template <typename T>
void min_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = std::min(dest[i], src[i]);
  }
}
/** Basic max reduction. */
template <typename T>
void max_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = std::max(dest[i], src[i]);
  }
}
/** Basic logical OR reduction. */
template <typename T>
void lor_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] || dest[i];
  }
}
/** Basic logical AND reduction. */
template <typename T>
void land_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] && dest[i];
  }
}
/** Basic logical XOR reduction. */
template <typename T>
void lxor_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = !src[i] != !dest[i];
  }
}
/** Basic bitwise OR reduction. */
template <typename T>
void bor_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] | dest[i];
  }
}
/** Basic bitwise AND reduction. */
template <typename T>
void band_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] & dest[i];
  }
}
/** Basic bitwise XOR reduction. */
template <typename T>
void bxor_reduction(const T* src, T* dest, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] ^ dest[i];
  }
}
// Binary operations are not supported on floating point types.
template <>
inline void bor_reduction<float>(const float*, float*, size_t) {
  throw_al_exception("BOR not supported for float");
}
template <>
inline void band_reduction<float>(const float*, float*, size_t) {
  throw_al_exception("BAND not supported for float");
}
template <>
inline void bxor_reduction<float>(const float*, float*, size_t) {
  throw_al_exception("BXOR not supported for float");
}
template <>
inline void bor_reduction<double>(const double*, double*, size_t) {
  throw_al_exception("BOR not supported for double");
}
template <>
inline void band_reduction<double>(const double*, double*, size_t) {
  throw_al_exception("BAND not supported for double");
}
template <>
inline void bxor_reduction<double>(const double*, double*, size_t) {
  throw_al_exception("BXOR not supported for double");
}
template <>
inline void bor_reduction<long double>(const long double*, long double*, size_t) {
  throw_al_exception("BOR not supported for long double");
}
template <>
inline void band_reduction<long double>(const long double*, long double*, size_t) {
  throw_al_exception("BAND not supported for long double");
}
template <>
inline void bxor_reduction<long double>(const long double*, long double*, size_t) {
  throw_al_exception("BXOR not supported for long double");
}

/** Return the associated reduction function for an operator. */
template <typename T>
inline std::function<void(const T*, T*, size_t)> ReductionMap(
  ReductionOperator op) {
  switch (op) {
  case ReductionOperator::sum:
    return sum_reduction<T>;
  case ReductionOperator::prod:
    return prod_reduction<T>;
  case ReductionOperator::min:
    return min_reduction<T>;
  case ReductionOperator::max:
    return max_reduction<T>;
  case ReductionOperator::lor:
    return lor_reduction<T>;
  case ReductionOperator::land:
    return land_reduction<T>;
  case ReductionOperator::lxor:
    return lxor_reduction<T>;
  case ReductionOperator::bor:
    return bor_reduction<T>;
  case ReductionOperator::band:
    return band_reduction<T>;
  case ReductionOperator::bxor:
    return bxor_reduction<T>;
  default:
    throw_al_exception("Reduction operator not supported");
  }
}

/** Convert a ReductionOperator to the corresponding MPI_Op. */
inline MPI_Op ReductionOperator2MPI_Op(ReductionOperator op) {
  switch (op) {
  case ReductionOperator::sum:
    return MPI_SUM;
  case ReductionOperator::prod:
    return MPI_PROD;
  case ReductionOperator::min:
    return MPI_MIN;
  case ReductionOperator::max:
    return MPI_MAX;
  case ReductionOperator::lor:
    return MPI_LOR;
  case ReductionOperator::land:
    return MPI_LAND;
  case ReductionOperator::lxor:
    return MPI_LXOR;
  case ReductionOperator::bor:
    return MPI_BOR;
  case ReductionOperator::band:
    return MPI_BAND;
  case ReductionOperator::bxor:
    return MPI_BXOR;
  default:
    throw_al_exception("Reduction operator not supported");
  }
}

}  // namespace mpi
}  // namespace internal
}  // namespace Al
