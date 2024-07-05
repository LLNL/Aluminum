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

#include <functional>
#include <vector>

#include <mpi.h>

#include <Al_config.hpp>
#include "aluminum/base.hpp"
#include "aluminum/internal.hpp"
#include "aluminum/datatypes.hpp"

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
#ifdef AL_HAS_HALF
// We use short as a dummy two-byte type.
// This is dispatched to special reduction operators when needed.
template <> inline MPI_Datatype TypeMap<__half>() { return MPI_SHORT; }
#endif
#ifdef AL_HAS_BFLOAT
// Same as for half.
template <> inline MPI_Datatype TypeMap<al_bfloat16>() { return MPI_SHORT; }
#endif


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

// Vector types for handling MPI counts and displacements.
// In the public Aluminum API, these are std::vector<size_t>.
// However, MPI uses either arrays of ints (standard calls) or arrays
// of MPI_Count (counts) or MPI_Aint (displacements), where MPI_Count
// and MPI_Aint are not necessarily the same type (large-count calls).
// These let us duck-type around this.

#ifdef AL_HAS_LARGE_COUNT_MPI
using Al_mpi_count_t = MPI_Count;
using Al_mpi_displ_t = MPI_Aint;
#else
using Al_mpi_count_t = int;
using Al_mpi_displ_t = int;
#endif
using Al_mpi_count_vector_t = std::vector<Al_mpi_count_t>;
using Al_mpi_displ_vector_t = std::vector<Al_mpi_displ_t>;

/** True if count elements can be sent by MPI. */
inline bool check_count_fits_mpi(size_t count) {
  return
    count <= static_cast<size_t>(std::numeric_limits<Al_mpi_count_t>::max());
}

/** Throw an exception if count elements cannot be sent by MPI. */
inline void assert_count_fits_mpi(size_t count) {
  if (!check_count_fits_mpi(count)) {
    throw_al_exception("Message count too large for MPI");
  }
}

/** True if displ is a valid MPI displacement. */
inline bool check_displ_fits_mpi(size_t displ) {
  return
    displ <= static_cast<size_t>(std::numeric_limits<Al_mpi_displ_t>::max());
}

/** Throw an exception if displ is not a valid MPI displacement. */
inline void assert_displ_fits_mpi(size_t displ) {
  if (!check_displ_fits_mpi(displ)) {
    throw_al_exception("Message displacement too large for MPI");
  }
}

/**
 * Convert a vector of size_ts to a vector of MPI counts.
 */
inline Al_mpi_count_vector_t
countify_size_t_vector(const std::vector<size_t>& v) {
#ifdef AL_DEBUG
  Al_mpi_count_vector_t count_v(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    assert_count_fits_mpi(v[i]);
    count_v[i] = v[i];
  }
  return count_v;
#else
  return Al_mpi_count_vector_t(v.begin(), v.end());
#endif
}

/**
 * Convert a vector of size_ts to a vector of MPI displacements.
 */
inline Al_mpi_displ_vector_t
displify_size_t_vector(const std::vector<size_t>& v) {
#ifdef AL_DEBUG
  Al_mpi_displ_vector_t displ_v(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    assert_displ_fits_mpi(v[i]);
    displ_v[i] = v[i];
  }
  return displ_v;
#else
  return Al_mpi_displ_vector_t(v.begin(), v.end());
#endif
}

/**
 * Call either a regular MPI function or the large-count version
 * depending on whether MPI supports the latter. This assumes that
 * the caller has the count argument in an appropriate type.
 */
#ifdef AL_HAS_LARGE_COUNT_MPI
#define AL_MPI_LARGE_COUNT_CALL(mpi_func) mpi_func##_c
#else
#define AL_MPI_LARGE_COUNT_CALL(mpi_func) mpi_func
#endif

#ifdef AL_HAS_HALF
/** Sum operator for half. */
extern MPI_Op half_sum_op;
/** Product operator for half. */
extern MPI_Op half_prod_op;
/** Min operator for half. */
extern MPI_Op half_min_op;
/** Max operator for half. */
extern MPI_Op half_max_op;
#endif

#ifdef AL_HAS_BFLOAT
/** Sum operator for bfloat. */
extern MPI_Op bfloat_sum_op;
/** Product operator for bfloat. */
extern MPI_Op bfloat_prod_op;
/** Min operator for bfloat. */
extern MPI_Op bfloat_min_op;
/** Max operator for bfloat. */
extern MPI_Op bfloat_max_op;
#endif

/** Convert a ReductionOperator to the corresponding MPI_Op. */
template <typename T>
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

#ifdef AL_HAS_HALF
template <>
inline MPI_Op ReductionOperator2MPI_Op<__half>(ReductionOperator op) {
  switch (op) {
  case ReductionOperator::sum:
    return half_sum_op;
  case ReductionOperator::prod:
    return half_prod_op;
  case ReductionOperator::min:
    return half_min_op;
  case ReductionOperator::max:
    return half_max_op;
  default:
    throw_al_exception("Reduction operator not supported");
  }
}
#endif

#ifdef AL_HAS_BFLOAT
template <>
inline MPI_Op ReductionOperator2MPI_Op<al_bfloat16>(ReductionOperator op) {
  switch (op) {
  case ReductionOperator::sum:
    return bfloat_sum_op;
  case ReductionOperator::prod:
    return bfloat_prod_op;
  case ReductionOperator::min:
    return bfloat_min_op;
  case ReductionOperator::max:
    return bfloat_max_op;
  default:
    throw_al_exception("Reduction operator not supported");
  }
}
#endif

}  // namespace mpi
}  // namespace internal
}  // namespace Al
