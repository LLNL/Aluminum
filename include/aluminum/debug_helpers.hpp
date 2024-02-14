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

/**
 * @file
 * Provide various utilities for debugging.
 */

#pragma once

#include <Al_config.hpp>

#include <stddef.h>
#include <stdint.h>
#include <numeric>
#include <vector>

#include "aluminum/base.hpp"
#include "base.hpp"

namespace Al {
namespace debug {

/**
 * Check whether buf is sane.
 *
 * Does nothing when AL_DEBUG is not defined.
 *
 * Current checks:
 * - buf is non-null or size is 0.
 */
template <typename T>
void check_buffer(const T* buf, size_t size) {
#ifdef AL_DEBUG
  if (size > 0 && buf == nullptr) {
    throw_al_exception("Null buffer");
  }
#else //  AL_DEBUG
  (void) buf;
  (void) size;
#endif  // AL_DEBUG
}

/**
 * Check whether buf is sane, but allow it to be null on the root.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename Backend, typename T>
void check_buffer_nonroot(const T* buf, size_t size, int root,
                          const typename Backend::comm_type& comm) {
#ifdef AL_DEBUG
  if (comm.rank() != root) {
    if (size > 0 && buf == nullptr) {
      throw_al_exception("Null buffer on nonroot");
    }
  }
#else  // AL_DEBUG
  (void) buf;
  (void) size;
  (void) root;
  (void) comm;
#endif  // AL_DEBUG
}

/**
 * Check whether buf is sane, but allow it to be null on non-roots.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename Backend, typename T>
void check_buffer_root(const T* buf, size_t size, int root,
                       const typename Backend::comm_type& comm) {
  #ifdef AL_DEBUG
  if (comm.rank() == root) {
    if (size > 0 && buf == nullptr) {
      throw_al_exception("Null buffer on root");
    }
  }
#else  // AL_DEBUG
  (void) buf;
  (void) size;
  (void) root;
  (void) comm;
#endif  // AL_DEBUG
}

/**
 * Check whether buf1 and buf2 overlap, throwing if they do.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename T>
void check_overlap(const T* buf1, size_t size1, const T* buf2, size_t size2) {
#ifdef AL_DEBUG
  // We work with integers because arithmetic with pointers is generally UB.
  uintptr_t b1 = reinterpret_cast<uintptr_t>(buf1);
  size_t s1 = sizeof(T) * size1;
  uintptr_t b2 = reinterpret_cast<uintptr_t>(buf2);
  size_t s2 = sizeof(T) * size2;

  if ((b1 < b2 + s2) && (b2 < b1 + s1)) {
    throw_al_exception("Overlapping buffers are not permitted");
  }
#else  // AL_DEBUG
  (void) buf1;
  (void) size1;
  (void) buf2;
  (void) size2;
#endif  // AL_DEBUG
}

/**
 * Check that a vector has size equal to a communicator's size.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename Backend, typename T>
void check_vector_is_comm_sized(const std::vector<T>& v,
                                const typename Backend::comm_type& comm) {
#ifdef AL_DEBUG
  if (v.size() < static_cast<size_t>(comm.size())) {
    throw_al_exception("Vector size (", v.size(),
                       ") does not equal communicator size (",
                       comm.size(), ")");
  }
#else  // AL_DEBUG
  (void) v;
  (void) comm;
#endif  // AL_DEBUG
}

/**
 * Return the entry in a vector corresponding to this processes's rank.
 *
 * Returns 0 when AL_DEBUG is not defined.
 */
template <typename Backend, typename T>
T get_rank_entry(const std::vector<T>& v, const typename Backend::comm_type& comm) {
#ifdef AL_DEBUG
  if (v.size() < static_cast<size_t>(comm.rank())) {
    throw_al_exception("Attempt to access entry ", comm.rank(),
                       " in vector of size ", v.size());
  }
  return v[comm.rank()];
#else  // AL_DEBUG
  (void) v;
  (void) comm;
  return T{0};
#endif  // AL_DEBUG
}

/**
 * Compute a sum of the elements in a vector, but only in debug mode.
 *
 * Returns 0 when AL_DEBUG is not defined.
 */
template <typename T, typename AccT = T>
AccT sum(const std::vector<T>& v) {
#ifdef AL_DEBUG
  return std::accumulate(v.cbegin(), v.cend(), AccT{0});
#else  // AL_DEBUG
  (void) v;
  return AccT{0};
#endif  // AL_DEBUG
}

/**
 * Check whether a rank is contained with in a communicator.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename Backend>
void check_rank(int rank, const typename Backend::comm_type& comm) {
#ifdef AL_DEBUG
  if (rank < 0 || rank >= comm.size()) {
    throw_al_exception("Rank ", rank, " is not in communicator");
  }
#else  // AL_DEBUG
  (void) rank;
  (void) comm;
#endif  // AL_DEBUG
}

/**
 * Check arguments for an Al::MultiSendRecv.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename Backend, typename T>
void check_multisendrecv(const std::vector<const T*>& send_buffers,
                         const std::vector<size_t>& send_counts,
                         const std::vector<int>& dests,
                         const std::vector<T*>& recv_buffers,
                         const std::vector<size_t>& recv_counts,
                         const std::vector<int>& srcs,
                         const typename Backend::comm_type& comm) {
#ifdef AL_DEBUG
  // TODO: Check buffers for overlap.
  if (send_buffers.size() != send_counts.size() ||
      send_buffers.size() != dests.size()) {
    throw_al_exception(
        "Send buffers, counts, and destinations must have the same length");
  }
  if (recv_buffers.size() != recv_counts.size() ||
      recv_buffers.size() != srcs.size()) {
    throw_al_exception(
        "Recv buffers, counts, and sources must have the same length");
  }
  for (size_t i = 0; i < dests.size(); ++i) {
    check_rank<Backend>(dests[i], comm);
    check_buffer(send_buffers[i], send_counts[i]);
  }
  for (size_t i = 0; i < srcs.size(); ++i) {
    check_rank<Backend>(srcs[i], comm);
    check_buffer(recv_buffers[i], recv_counts[i]);
  }
#else  // AL_DEBUG
  (void) send_buffers;
  (void) send_counts;
  (void) dests;
  (void) recv_buffers;
  (void) recv_counts;
  (void) srcs;
  (void) comm;
#endif  // AL_DEBUG
}

/**
 * Check arguments for an inplace Al::MultiSendRecv.
 *
 * Does nothing when AL_DEBUG is not defined.
 */
template <typename Backend, typename T>
void check_inplace_multisendrecv(const std::vector<T*>& buffers,
                                 const std::vector<size_t>& counts,
                                 const std::vector<int>& dests,
                                 const std::vector<int>& srcs,
                                 const typename Backend::comm_type& comm) {
#ifdef AL_DEBUG
  // TODO: Check buffers for overlap.
  if (buffers.size() != counts.size() || buffers.size() != dests.size() ||
      buffers.size() != srcs.size()) {
    throw_al_exception(
        "Buffers, counts, destinations, and sources must have the same length");
  }
  for (size_t i = 0; i < dests.size(); ++i) {
    check_rank<Backend>(dests[i], comm);
    check_rank<Backend>(srcs[i], comm);
    check_buffer(buffers[i], counts[i]);
  }
#else  // AL_DEBUG
  (void) buffers;
  (void) counts;
  (void) dests;
  (void) srcs;
  (void) comm;
#endif  // AL_DEBUG
}

}  // namespace debug
}  // namespace Al
