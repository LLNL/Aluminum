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

#include <algorithm>
#include <vector>
#include <numeric>

#include "internal.hpp"
#include "progress.hpp"
#include "mempool.hpp"

namespace Al {

namespace internal {
namespace mpi{
int get_max_tag();
}
}

/**
 * Communicator for MPI-based collectives.
 */
class MPICommunicator : public Communicator {
 public:
  /** Default constructor; use MPI_COMM_WORLD. */
  MPICommunicator() : MPICommunicator(MPI_COMM_WORLD) {}
  /** Use a particular MPI communicator. */
  MPICommunicator(MPI_Comm comm_) : Communicator() {
    // Duplicate the communicator to avoid interference.
    MPI_Comm_dup(comm_, &comm);
    MPI_Comm_rank(comm, &rank_in_comm);
    MPI_Comm_size(comm, &size_of_comm);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &local_comm);
    MPI_Comm_rank(local_comm, &rank_in_local_comm);
    MPI_Comm_size(local_comm, &size_of_local_comm);
  }
  virtual ~MPICommunicator() override {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
      // TODO: temporary skipping deleting comm
      //MPI_Comm_free(&comm);
      //MPI_Comm_free(&local_comm);
    }
  }
  Communicator* copy() const override { return new MPICommunicator(comm); }
  int rank() const override { return rank_in_comm; }
  int size() const override { return size_of_comm; }
  MPI_Comm get_comm() const { return comm; }
  int local_rank() const override { return rank_in_local_comm; }
  int local_size() const override { return size_of_local_comm; }
  MPI_Comm get_local_comm() const { return local_comm; }
  int get_free_tag() {
    int tag = free_tag++;
    if (free_tag >= internal::mpi::get_max_tag()
        || free_tag < starting_free_tag) {
      free_tag = 10;
    }
    return tag;
  }

 private:
  /** Associated MPI communicator. */
  MPI_Comm comm;
  /** Rank in comm. */
  int rank_in_comm;
  /** Size of comm. */
  int size_of_comm;
  /** Communicator for the local node. */
  MPI_Comm local_comm;
  /** Rank in the local communicator. */
  int rank_in_local_comm;
  /** Size of the local communicator. */
  int size_of_local_comm;
  /**
   * Starting tag to use for non-blocking operations.
   * No other operations should use any tag >= to this one.
   */
  static constexpr int starting_free_tag = 10;
  /** Free tag for communication. */
  int free_tag = starting_free_tag;
};

namespace internal {
namespace mpi {

/** MPI initialization. */
void init(int& argc, char**& argv);
/** MPI finalization. */
void finalize();

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
template <> inline MPI_Datatype TypeMap<long int>() { return MPI_LONG_INT; }
template <> inline MPI_Datatype TypeMap<unsigned long int>() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype TypeMap<long long int>() { return MPI_LONG_LONG_INT; }
template <> inline MPI_Datatype TypeMap<unsigned long long int>() { return MPI_UNSIGNED_LONG_LONG; }
template <> inline MPI_Datatype TypeMap<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype TypeMap<double>() { return MPI_DOUBLE; }
template <> inline MPI_Datatype TypeMap<long double>() { return MPI_LONG_DOUBLE; }

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
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_SUM_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] += src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] += src[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] += src[i];
  }
#endif
}
/** Basic prod reduction. */
template <typename T>
void prod_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_PROD_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] *= src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] *= src[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] *= src[i];
  }
#endif
}
/** Basic min reduction. */
template <typename T>
void min_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_MINMAX_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::min(dest[i], src[i]);
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::min(dest[i], src[i]);
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = std::min(dest[i], src[i]);
  }
#endif
}
/** Basic max reduction. */
template <typename T>
void max_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_MINMAX_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::max(dest[i], src[i]);
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::max(dest[i], src[i]);
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = std::max(dest[i], src[i]);
  }
#endif
}
/** Basic logical OR reduction. */
template <typename T>
void lor_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_LOGICAL_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] || dest[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] || dest[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] || dest[i];
  }
#endif
}
/** Basic logical AND reduction. */
template <typename T>
void land_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_LOGICAL_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] && dest[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] && dest[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] && dest[i];
  }
#endif
}
/** Basic logical XOR reduction. */
template <typename T>
void lxor_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_LOGICAL_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = !src[i] != !dest[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = !src[i] != !dest[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = !src[i] != !dest[i];
  }
#endif
}
/** Basic bitwise OR reduction. */
template <typename T>
void bor_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_BITWISE_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] | dest[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] | dest[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] | dest[i];
  }
#endif
}
/** Basic bitwise AND reduction. */
template <typename T>
void band_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_BITWISE_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] & dest[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] & dest[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] & dest[i];
  }
#endif
}
/** Basic bitwise XOR reduction. */
template <typename T>
void bxor_reduction(const T* src, T* dest, size_t count) {
#if AL_MPI_USE_OPENMP
  if (count >= AL_MPI_MULTITHREAD_BITWISE_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] ^ dest[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = src[i] ^ dest[i];
    }
  }
#else
  for (size_t i = 0; i < count; ++i) {
    dest[i] = src[i] ^ dest[i];
  }
#endif
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

/** Default tag for blocking operations. */
constexpr int default_tag = 0;

/** Base state class for MPI allreduces. */
template <typename T>
class MPIAlState : public AlState {
 public:
  MPIAlState(const T* sendbuf_, T* recvbuf_, size_t count_,
             ReductionOperator op_, Communicator& comm_,
             AlRequest req_) :
    AlState(req_), sendbuf(sendbuf_), recvbuf(recvbuf_), recv_to(nullptr),
    count(count_) {
    comm = dynamic_cast<MPICommunicator&>(comm_).get_comm();
    type = TypeMap<T>();
    reduction_op = ReductionMap<T>(op_);
    rank = comm_.rank();
    nprocs = comm_.size();
    tag = dynamic_cast<MPICommunicator&>(comm_).get_free_tag();
  }
  ~MPIAlState() override {
    if (recv_to != nullptr) {
      release_memory(recv_to);
    }
  }
  /**
   * Performs generic setup and checks.
   * If the allreduce is completed immediately, returns true.
   */
  virtual bool setup() {
    if (count == 0) {
      return true;  // Nothing to do.
    }
    assert_count_fits_mpi(count);
    if (sendbuf != IN_PLACE<T>()) {
      // Copy data to the receive buffer.
      std::copy_n(sendbuf, count, recvbuf);
    }
    if (nprocs == 1) {
      return true;  // Only needed to copy data.
    }
    return false;
  }
 protected:
  /** Local rank. */
  int rank;
  /** Number of processes in communicator. */
  int nprocs;
  /** Buffer to send from */
  const T* sendbuf;
  /** Buffer to receive to. */
  T* recvbuf;
  /** Temporary buffer for intermediate data. */
  T* recv_to;
  /** Number of elements in sendbuf/recvbuf. */
  size_t count;
  /** Communicator for the allreduce. */
  MPI_Comm comm;
  /** MPI datatype for T. */
  MPI_Datatype type;
  /** Reduction operator to use. */
  std::function<void(const T*, T*, size_t)> reduction_op;
  /**
   * Tag to use for MPI operations.
   * This is selected to avoid interference with other allreduces.
   */
  int tag;
  /** Requests for send_recv. */
  MPI_Request send_recv_reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
#ifdef AL_DEBUG_HANG_CHECK
  bool hang_reported = false;
  double send_recv_start = std::numeric_limits<double>::max();
#endif

  /** Start a send/recv. */
  void start_send_recv(
    const void* send, int send_count, int dest,
    void* recv, int recv_count, int source) {
    MPI_Irecv(recv, recv_count, type, source, tag, comm, &(send_recv_reqs[0]));
    MPI_Isend(send, send_count, type, dest, tag, comm, &(send_recv_reqs[1]));
#ifdef AL_DEBUG_HANG_CHECK
    send_recv_start = get_time();
#endif
  }
  /** Return true if the outstanding send/recv has completed. */
  bool test_send_recv() {
    int flag;
    MPI_Testall(2, send_recv_reqs, &flag, MPI_STATUSES_IGNORE);
#ifdef AL_DEBUG_HANG_CHECK
    if (!flag && !hang_reported) {
      double t = get_time();
      // Choice of 10 + rank is arbitrary, but seems reasonable.
      // The + rank part helps ensure printing from ranks isn't interleaved.
      if (t - send_recv_start > 10.0 + rank) {
        std::cout << rank << ": Possible send/recv hang detected, tag=" << tag << std::endl;
        hang_reported = true;
      }
    }
#endif
    return flag;
  }
};

/** Just call MPI_Allreduce directly. */
template <typename T>
void passthrough_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                           ReductionOperator op, Communicator& comm) {
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  MPI_Datatype type = TypeMap<T>();
  MPI_Op mpi_op = ReductionOperator2MPI_Op(op);
  if (sendbuf == IN_PLACE<T>()) {
    MPI_Allreduce(MPI_IN_PLACE, recvbuf, count, type, mpi_op, mpi_comm);
  } else {
    MPI_Allreduce(sendbuf, recvbuf, count, type, mpi_op, mpi_comm);
  }
}

template <typename T>
class MPIPassthroughAlState : public MPIAlState<T> {
 public:
  MPIPassthroughAlState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AlRequest req_) :
    MPIAlState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {
    mpi_op = ReductionOperator2MPI_Op(op_);
  }
  bool setup() override {
    // Don't need to call the parent. Just start the MPI allreduce.
    if (this->sendbuf == IN_PLACE<T>()) {
      MPI_Iallreduce(MPI_IN_PLACE, this->recvbuf, this->count, this->type,
                     mpi_op, this->comm, &mpi_req);
    } else {
      MPI_Iallreduce(this->sendbuf, this->recvbuf, this->count, this->type,
                     mpi_op, this->comm, &mpi_req);
    }
    return false;
  }
  bool step() override {
    int flag;
    MPI_Test(&mpi_req, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      return true;
    }
    return false;
  }
 private:
  MPI_Op mpi_op;
  MPI_Request mpi_req;
};

/** Just call MPI_Iallreduce directly. */
template <typename T>
void nb_passthrough_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                              ReductionOperator op, Communicator& comm,
                              AlRequest& req) {
  req = get_free_request();
  MPIPassthroughAlState<T>* state =
    new MPIPassthroughAlState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  state->setup();
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

/** Use a recursive-doubling algorithm to perform the allreduce. */
template <typename T>
void recursive_doubling_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                  ReductionOperator op, Communicator& comm,
                                  const int tag = default_tag) {
  if (count == 0) return;  // Nothing to do.
  assert_count_fits_mpi(count);
  int rank = comm.rank();
  int nprocs = comm.size();
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  if (sendbuf != IN_PLACE<T>()) {
    // Copy our data into the receive buffer.
    std::copy_n(sendbuf, count, recvbuf);
  }
  if (nprocs == 1) return;  // Only needed to copy data.
  // TODO: Shared memory optimization.
  MPI_Datatype type = TypeMap<T>();
  unsigned int mask = 1;
  T* recv_to = get_memory<T>(count);
  auto reduction_op = ReductionMap<T>(op);
  // Check if we are in a non-power-of-2 case.
  // First find the nearest power-of-2 <= nprocs.
  int pow2 = 1;
  while (pow2 <= nprocs) pow2 <<= 1;
  pow2 >>= 1;
  int pow2_remainder = nprocs - pow2;
  int orig_rank = rank;
  if (rank < 2 * pow2_remainder) {
    // We have a non-power-of-2 number of processes (pow2_remainder > 0).
    // The *even* processes of rank < 2*pow2_remainder will send their data
    // to the odd processes, which will reduce it into their data.
    // The even processes will not participate until the end.
    // There is a power-of-2 number of remaining processes.
    if (rank % 2 == 0) {
      MPI_Send(recvbuf, count, type, rank + 1, tag, mpi_comm);
      rank = -1;  // Don't participate.
    } else {
      MPI_Recv(recv_to, count, type, rank - 1, tag, mpi_comm, MPI_STATUS_IGNORE);
      reduction_op(recv_to, recvbuf, count);
      rank /= 2;  // Change our rank.
    }
  } else {
    rank -= pow2_remainder;  // This is a NOP when nprocs is power-of-2.
  }
  while (rank != -1 && mask < static_cast<unsigned int>(pow2)) {
    // Need to get real rank.
    int adjusted_partner = rank ^ mask;
    int partner = (adjusted_partner < pow2_remainder) ?
      adjusted_partner * 2 + 1 :
      adjusted_partner + pow2_remainder;
    MPI_Sendrecv(recvbuf, count, type, partner, tag,
                 recv_to, count, type, partner, tag,
                 mpi_comm, MPI_STATUS_IGNORE);
    reduction_op(recv_to, recvbuf, count);
    mask <<= 1;
  }
  // In the non-power-of-2 case, the even ranks need to get their data.
  if (orig_rank < 2 * pow2_remainder) {
    if (orig_rank % 2 == 0) {
      MPI_Recv(recvbuf, count, type, orig_rank + 1, tag, mpi_comm,
               MPI_STATUS_IGNORE);
    } else {
      MPI_Send(recvbuf, count, type, orig_rank - 1, tag, mpi_comm);
    }
  }
  release_memory(recv_to);
}

template <typename T>
class MPIRecursiveDoublingAlState : public MPIAlState<T> {
 public:
  MPIRecursiveDoublingAlState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AlRequest req_) :
    MPIAlState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {}
  bool setup() override {
    bool r = MPIAlState<T>::setup();
    if (!r) {
      this->recv_to = get_memory<T>(this->count);
      // Check if we're in a non-power-of-2 case.
      while (pow2 <= this->nprocs) pow2 <<= 1;
      pow2 >>= 1;
      pow2_remainder = this->nprocs - pow2;
      adjusted_rank = this->rank;
      // Adjust rank and start a send/recv for the data if needed.
      if (this->rank < 2 * pow2_remainder) {
        if (this->rank % 2 == 0) {
          MPI_Isend(this->recvbuf, this->count, this->type, this->rank + 1,
                    this->tag, this->comm, &(this->send_recv_reqs[0]));
          adjusted_rank = -1;  // Don't participate.
        } else {
          MPI_Irecv(this->recv_to, this->count, this->type, this->rank - 1,
                    this->tag, this->comm, &(this->send_recv_reqs[0]));
          adjusted_rank /= 2;
        }
      } else {
        setup_comm_done = true;  // No send/recv on this process.
        adjusted_rank -= pow2_remainder;
      }
    }
    return r;
  }
  bool step() override {
    // Check the send/recv from setup, if any.
    if (!setup_comm_done) {
      if (this->test_send_recv()) {
        setup_comm_done = true;
        if (this->rank % 2) {
          // We received, need to reduce the data into our local buffer.
          this->reduction_op(this->recv_to, this->recvbuf, this->count);
        }
      } else {
        return false;
      }
    }
    // Complete final communication in the non-power-of-2 case.
    if (final_comm_started) {
      return this->test_send_recv();
    }
    if (adjusted_rank == -1) {
      // Just need to wait for data.
      MPI_Irecv(this->recvbuf, this->count, this->type, this->rank + 1,
                this->tag, this->comm, &(this->send_recv_reqs[0]));
      final_comm_started = true;
      return false;
    }
    bool test = this->test_send_recv();
    if (started && test) {
      // Send completed, reduce and update state.
      this->reduction_op(this->recv_to, this->recvbuf, this->count);
      mask <<= 1;
      if (mask >= static_cast<unsigned int>(pow2)) {
        // Done, but in the non-power-of-2 case we need to send to our partner.
        if (this->rank < 2 * pow2_remainder) {
          MPI_Isend(this->recvbuf, this->count, this->type, this->rank - 1,
                    this->tag, this->comm, &(this->send_recv_reqs[0]));
          final_comm_started = true;
          return false;
        } else {
          return true;
        }
      }
    }
    if (test) {
      // Compute the real rank to send to.
      const int adjusted_partner = this->adjusted_rank ^ mask;
      const int partner = (adjusted_partner < pow2_remainder) ?
        adjusted_partner * 2 + 1 :
        adjusted_partner + pow2_remainder;
      this->start_send_recv(this->recvbuf, this->count, partner,
                            this->recv_to, this->count, partner);
      started = true;
    }
    return false;
  }
 private:
  /** Mask for computing partners in each step. */
  unsigned int mask = 1;
  /** Whether communication has started. */
  bool started = false;
  /** Whether the send/recv from non-power-of-2 setup has completed. */
  bool setup_comm_done = false;
  /** Whether the final send/recv from the non-power-of-2 case has started. */
  bool final_comm_started = false;
  /** Nearest power-of-2 <= nprocs. */
  int pow2 = 1;
  /** Processes left over in the non-power-of-2 case. */
  int pow2_remainder = 0;
  /**
   * Process's adjusted rank for the non-power-of-2 case.
   * This is equal to rank if nprocs is a power of 2.
   */
  int adjusted_rank = -1;
};

/** Non-blocking recursive-doubling allreduce. */
template <typename T>
void nb_recursive_doubling_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                     ReductionOperator op, Communicator& comm,
                                     AlRequest& req) {
  req = get_free_request();
  MPIRecursiveDoublingAlState<T>* state =
    new MPIRecursiveDoublingAlState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  if (state->setup()) {
    req = NULL_REQUEST;
    return;
  }
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

/** Use a ring-based reduce-scatter then allgather to perform the allreduce. */
template <typename T>
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, Communicator& comm, const bool bidir,
                    const size_t num_rings, const int tag = default_tag) {
  if (count == 0) return;  // Nothing to do.
  if (num_rings != 1) {
    throw_al_exception("Multiple rings not yet supported");
  }
  const size_t bidir_cnt = bidir ? 2 : 1;
  assert_count_fits_mpi(count);
  int rank = comm.rank();
  int nprocs = comm.size();
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  if (sendbuf != IN_PLACE<T>()) {
    // Copy our data into the receive buffer.
    std::copy_n(sendbuf, count, recvbuf);
  }
  if (nprocs == 1) return;  // Only needed to copy data.
  // Compute the slices of data to be moved in each ring.
  const size_t num_slices = nprocs * num_rings * bidir_cnt;
  const size_t size_per_msg = count / num_slices;
  size_t remainder = count % num_slices;
  // Indexing: [ring][dir][proc]
  std::vector<std::vector<std::vector<size_t>>> slice_lengths(num_rings);
  std::vector<std::vector<std::vector<size_t>>> slice_ends(num_rings);
  size_t cur_len = 0;
  for (size_t i = 0; i < num_rings; ++i) {
    slice_lengths[i].resize(bidir_cnt);
    slice_ends[i].resize(bidir_cnt);
    for (size_t j = 0; j < bidir_cnt; ++j) {
      slice_lengths[i][j].resize(nprocs, size_per_msg);
      // Add in the remainder as needed.
      const size_t to = std::min(static_cast<size_t>(nprocs), remainder);
      for (size_t k = 0; k < to; ++k) {
        slice_lengths[i][j][k] += 1;
      }
      remainder -= to;
      slice_ends[i][j].resize(nprocs);
      // Compute the end location of each slice, in absolute terms.
      for (size_t k = 0; k < static_cast<size_t>(nprocs); ++k) {
        cur_len += slice_lengths[i][j][k];
        slice_ends[i][j][k] = cur_len;
      }
    }
  }
  // Temporary buffers for receiving. Indexing: [ring][dir]
  std::vector<std::vector<T*>> recv_to(num_rings);
  for (size_t i = 0; i < num_rings; ++i) {
    recv_to[i].resize(bidir_cnt);
    for (size_t j = 0; j < bidir_cnt; ++j) {
      recv_to[i][j] = get_memory<T>(slice_lengths[0][0][0]);
    }
  }
  // Compute sources (left) and destinations (right) for the rings.
  // Indexing: [ring][dir]
  // TODO: Compute ranks for num_rings > 1.
  std::vector<std::vector<int>> srcs(num_rings);
  for (size_t i = 0; i < num_rings; ++i) srcs[i].resize(bidir_cnt);
  srcs[0][0] = (rank - 1 + nprocs) % nprocs;
  std::vector<std::vector<int>> dsts(num_rings);
  for (size_t i = 0; i < num_rings; ++i) dsts[i].resize(bidir_cnt);
  dsts[0][0] = (rank + 1) % nprocs;
  if (bidir) {
    srcs[0][1] = (rank + 1) % nprocs;
    dsts[0][1] = (rank - 1 + nprocs) % nprocs;
  }
  // This is used to determine how the slices are moved through based on the
  // direction and step.
  const int step_direction[2] = {-1, 1};
  MPI_Datatype type = TypeMap<T>();
  auto reduction_op = ReductionMap<T>(op);
  // TODO: Merge RS/AG steps so the RS doesn't have to complete on every ring
  // before the AG starts.
  // Ring reduce-scatter.
  const size_t num_transfers = num_rings * bidir_cnt * (nprocs - 1);
  size_t completed_transfers = 0;
  // Needs to be contiguous.
  std::vector<MPI_Request> reqs(num_rings * bidir_cnt * 2, MPI_REQUEST_NULL);
  // Current step of each ring/direction.
  std::vector<std::vector<int>> steps(num_rings);
  for (size_t i = 0; i < num_rings; ++i) steps[i].resize(bidir_cnt, 0);
  // Start the first transfers.
  for (size_t ring = 0; ring < num_rings; ++ring) {
    for (size_t dir = 0; dir < bidir_cnt; ++dir) {
      size_t req_idx = ring*num_rings + dir*2;
      // Determine the slices being sent and received.
      const int send_idx = rank;
      const int recv_idx = (rank + step_direction[dir] + nprocs) % nprocs;
      const T* to_send = recvbuf + (slice_ends[ring][dir][send_idx] -
                                    slice_lengths[ring][dir][send_idx]);
      MPI_Irecv(recv_to[ring][dir], slice_lengths[ring][dir][recv_idx],
                type, srcs[ring][dir], tag, mpi_comm, &(reqs[req_idx]));
      MPI_Isend(to_send, slice_lengths[ring][dir][send_idx], type,
                dsts[ring][dir], tag, mpi_comm, &(reqs[req_idx+1]));
    }
  }
  while (completed_transfers < num_transfers) {
    int idx;
    MPI_Waitany(reqs.size(), reqs.data(), &idx, MPI_STATUS_IGNORE);
    // Determine what completed.
    const int ring = idx / (bidir_cnt*2);
    const int dir = (idx - ring*num_rings) / 2;
    const int sendrecv = idx % 2;  // Whether this was a send or a receive.
    const int req_idx = ring*num_rings + dir*2;
    if (sendrecv == 0) {
      // Recv completed.
      const int recv_idx =
        (rank + step_direction[dir]*(steps[ring][dir] + 1) + nprocs) % nprocs;
      reduction_op(recv_to[ring][dir], recvbuf +
                   (slice_ends[ring][dir][recv_idx] -
                    slice_lengths[ring][dir][recv_idx]),
                   slice_lengths[ring][dir][recv_idx]);
    }
    // Check if this step is done.
    if (reqs[req_idx] == MPI_REQUEST_NULL &&
        reqs[req_idx+1] == MPI_REQUEST_NULL) {
      // Update counters.
      ++completed_transfers;
      ++steps[ring][dir];
      if (steps[ring][dir] < nprocs - 1) {
        // Start next set of transfers.
        const int send_idx =
          (rank + step_direction[dir]*steps[ring][dir] + nprocs) % nprocs;
        const int recv_idx =
          (rank + step_direction[dir]*(steps[ring][dir] + 1) + nprocs) % nprocs;
        const T* to_send = recvbuf + (slice_ends[ring][dir][send_idx] -
                                      slice_lengths[ring][dir][send_idx]);
        MPI_Irecv(recv_to[ring][dir], slice_lengths[ring][dir][recv_idx],
                  type, srcs[ring][dir], tag, mpi_comm, &(reqs[req_idx]));
        MPI_Isend(to_send, slice_lengths[ring][dir][send_idx], type,
                  dsts[ring][dir], tag, mpi_comm, &(reqs[req_idx+1]));
      }
    }
  }
  // Ring allgather.
  // Reset steps/transfers.
  completed_transfers = 0;
  for (size_t i = 0; i < num_rings; ++i) {
    for (size_t j = 0; j < bidir_cnt; ++j) {
      steps[i][j] = 0;
    }
  }
  // No temporary buffer is needed here: Receive directly into recvbuf.
  // Start the first transfers.
  for (size_t ring = 0; ring < num_rings; ++ring) {
    for (size_t dir = 0; dir < bidir_cnt; ++dir) {
      size_t req_idx = ring*num_rings + dir*2;
      //const int send_idx = (rank + 1) % nprocs;
      const int send_idx = (rank - step_direction[dir] + nprocs) % nprocs;
      const int recv_idx = rank;
      const T* to_send = recvbuf + (slice_ends[ring][dir][send_idx] -
                                    slice_lengths[ring][dir][send_idx]);
      MPI_Irecv(recvbuf + (slice_ends[ring][dir][recv_idx]
                           - slice_lengths[ring][dir][recv_idx]),
                slice_lengths[ring][dir][recv_idx],
                type, srcs[ring][dir], tag, mpi_comm, &(reqs[req_idx]));
      MPI_Isend(to_send, slice_lengths[ring][dir][send_idx], type,
                dsts[ring][dir], tag, mpi_comm, &(reqs[req_idx+1]));
    }
  }
  while (completed_transfers < num_transfers) {
    int idx;
    MPI_Waitany(reqs.size(), reqs.data(), &idx, MPI_STATUS_IGNORE);
    // Determine what completed.
    const int ring = idx / (bidir_cnt*2);
    const int dir = (idx - ring*num_rings) / 2;
    const int req_idx = ring*num_rings + dir*2;
    // Check if this step is done.
    if (reqs[req_idx] == MPI_REQUEST_NULL &&
        reqs[req_idx+1] == MPI_REQUEST_NULL) {
      // Update counters.
      ++completed_transfers;
      ++steps[ring][dir];
      if (steps[ring][dir] < nprocs - 1) {
        // Start next set of transfers.
        const int send_idx =
          (rank + step_direction[dir]*(steps[ring][dir] - 1) + nprocs) % nprocs;
        const int recv_idx =
          (rank + step_direction[dir]*steps[ring][dir] + nprocs) % nprocs;
        const T* to_send = recvbuf + (slice_ends[ring][dir][send_idx] -
                                      slice_lengths[ring][dir][send_idx]);
        MPI_Irecv(recvbuf + (slice_ends[ring][dir][recv_idx]
                             - slice_lengths[ring][dir][recv_idx]),
                  slice_lengths[ring][dir][recv_idx],
                  type, srcs[ring][dir], tag, mpi_comm, &(reqs[req_idx]));
        MPI_Isend(to_send, slice_lengths[ring][dir][send_idx], type,
                  dsts[ring][dir], tag, mpi_comm, &(reqs[req_idx+1]));
      }
    }
  }
  // Release temporary receive buffers.
  for (size_t i = 0; i < num_rings; ++i) {
    for (size_t j = 0; j < bidir_cnt; ++j) {
      release_memory(recv_to[i][j]);
    }
  }
}

template <typename T>
class MPIRingAlState : public MPIAlState<T> {
 public:
  MPIRingAlState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AlRequest req_) :
    MPIAlState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {}
  bool setup() override {
    bool r = MPIAlState<T>::setup();
    if (!r) {
      // Compute slices of data to be moved.
      const size_t size_per_rank = this->count / this->nprocs;
      const size_t remainder = this->count % this->nprocs;
      slice_lengths.resize(this->nprocs, size_per_rank);
      // Add in the remainder as evenly as possible.
      for (size_t i = 0; i < remainder; ++i) {
        slice_lengths[i] += 1;
      }
      slice_ends.resize(this->nprocs);
      std::partial_sum(slice_lengths.begin(), slice_lengths.end(),
                       slice_ends.begin());
      this->recv_to = get_memory<T>(slice_lengths[0]);
      src = (this->rank - 1 + this->nprocs) % this->nprocs;
      dst = (this->rank + 1) % this->nprocs;
      ag_send_idx = (this->rank + 1) % this->nprocs;
    }
    return r;
  }
  bool step() override {
    if (phase == 0) {
      if (rs_step()) {
        // Switch to allgather for next step.
        phase = 1;
        started = false;
        cur_step = 0;
      }
      return false;
    } else {
      return ag_step();
    }
  }
 private:
  /** 0 == reduce-scatter, 1 == allgather. */
  int phase = 0;
  /** Whether communication has started. */
  bool started = false;
  /** Current step in either reduce-scatter or all-gather. */
  int cur_step = 0;
  /** Left partner/source for the ring. */
  int src;
  /** Right partner/destination for the ring. */
  int dst;
  /** Send index for the allgather. */
  int ag_send_idx;
  /** Lengths of each slice of data to be sent. */
  std::vector<size_t> slice_lengths;
  /** End of each slice of data to be send. */
  std::vector<size_t> slice_ends;
  bool rs_step() {
    bool test = this->test_send_recv();
    if (started && test) {
      // Send completed, reduce and update state.
      const int old_recv_idx = (this->rank - cur_step - 1 + this->nprocs) %
        this->nprocs;
      this->reduction_op(
        this->recv_to,
        this->recvbuf +
          (slice_ends[old_recv_idx] - slice_lengths[old_recv_idx]),
        slice_lengths[old_recv_idx]);
      ++cur_step;
      if (cur_step >= this->nprocs - 1) {
        return true;
      }
    }
    if (test) {
      const int send_idx = (this->rank - cur_step + this->nprocs) %
        this->nprocs;
      const int recv_idx = (this->rank - cur_step - 1 + this->nprocs) %
        this->nprocs;
      const T* to_send = this->recvbuf +
        (slice_ends[send_idx] - slice_lengths[send_idx]);
      this->start_send_recv(to_send, slice_lengths[send_idx], dst,
                            this->recv_to, slice_lengths[recv_idx], src);
      started = true;
    }
    return false;
  }
  bool ag_step() {
    bool test = this->test_send_recv();
    if (started && test) {
      // Update state.
      const int old_recv_idx = (this->rank - cur_step + this->nprocs) % this->nprocs;
      ag_send_idx = old_recv_idx;
      ++cur_step;
      if (cur_step >= this->nprocs - 1) {
        return true;
      }
    }
    if (test) {
      const int recv_idx = (this->rank - cur_step + this->nprocs) % this->nprocs;
      const T* to_send =
        this->recvbuf + (slice_ends[ag_send_idx] - slice_lengths[ag_send_idx]);
      this->start_send_recv(
        to_send, slice_lengths[ag_send_idx], dst,
        this->recvbuf + (slice_ends[recv_idx] - slice_lengths[recv_idx]),
        slice_lengths[recv_idx], src);
      started = true;
    }
    return false;
  }
};

/** Non-blocking ring allreduce. */
template <typename T>
void nb_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, Communicator& comm,
                       AlRequest& req) {
  req = get_free_request();
  MPIRingAlState<T>* state =
    new MPIRingAlState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  if (state->setup()) {
    req = NULL_REQUEST;
    return;
  }
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

/**
 * Use Rabenseifner's algorithm (recursive halving/doubling) to perform the
 * allreduce.
 */
template <typename T>
void rabenseifner_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                            ReductionOperator op, Communicator& comm,
                            const int tag = default_tag) {
  if (count == 0) return;  // Nothing to do.
  assert_count_fits_mpi(count);
  int rank = comm.rank();
  int nprocs = comm.size();
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  if (sendbuf != IN_PLACE<T>()) {
    // Copy our data into the receive buffer.
    std::copy_n(sendbuf, count, recvbuf);
  }
  if (nprocs == 1) return;  // Only needed to copy data.
  MPI_Datatype type = TypeMap<T>();
  auto reduction_op = ReductionMap<T>(op);
  // Check if we are in the non-power-of-2 case.
  // This works basically as with recursive-doubling.
  int pow2 = 1;
  while (pow2 <= nprocs) pow2 <<= 1;
  pow2 >>= 1;
  int pow2_remainder = nprocs - pow2;
  int orig_rank = rank;
  // Temporary buffer for receiving. In the non-power-of-2 case, we need a
  // larger buffer; otherwise we will receive at most half the data.
  T* recv_to = nullptr;
  if (rank < 2 * pow2_remainder) {
    if (rank % 2 == 0) {
      MPI_Send(recvbuf, count, type, rank + 1, tag, mpi_comm);
      rank = -1;  // Don't participate.
    } else {
      recv_to = get_memory<T>(count);
      MPI_Recv(recv_to, count, type, rank - 1, tag, mpi_comm, MPI_STATUS_IGNORE);
      reduction_op(recv_to, recvbuf, count);
      rank /= 2;
    }
  } else {
    rank -= pow2_remainder;
  }
  if (rank != -1) {
    // Compute the slices of data to be moved.
    const size_t size_per_rank = count / pow2;
    const size_t remainder = count % pow2;
    std::vector<size_t> slice_lengths(pow2, size_per_rank);
    // Add in the remainder as evenly as possible.
    for (size_t i = 0; i < remainder; ++i) {
      slice_lengths[i] += 1;
    }
    std::vector<size_t> slice_ends(pow2);
    std::partial_sum(slice_lengths.begin(), slice_lengths.end(),
                     slice_ends.begin());
    // Allocate temporary buffer if needed.
    if (recv_to == nullptr) {
      recv_to = get_memory<T>(slice_ends[pow2 / 2]);
    }
    // Do a recursive-halving reduce-scatter.
    unsigned int partner_mask = pow2 >> 1;
    // Used to compute the number of slices sent in a step.
    unsigned int slice_mask = 1;
    int send_idx = 0;  // Starting index for sending.
    int recv_idx = 0;  // Starting index for receiving.
    int last_idx = pow2;  // End of right-most region.
    while (partner_mask > 0) {
      // Compute the real rank.
      int adjusted_partner = rank ^ partner_mask;
      int partner = (adjusted_partner < pow2_remainder) ?
        adjusted_partner * 2 + 1 :
        adjusted_partner + pow2_remainder;
      // Compute the range of data to send and receive.
      size_t send_start, send_end, recv_start, recv_end;
      // The check is done on the adjusted partner rank.
      if (rank < adjusted_partner) {
        send_idx = recv_idx + pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[last_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[send_idx - 1];
      } else {
        recv_idx = send_idx + pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[recv_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[last_idx - 1];
      }
      MPI_Sendrecv(recvbuf + send_start, send_end - send_start, type, partner, tag,
                   recv_to, recv_end - recv_start, type, partner, tag,
                   mpi_comm, MPI_STATUS_IGNORE);
      reduction_op(recv_to, recvbuf + recv_start, recv_end - recv_start);
      // Update for the next iteration, except last_idx, which is needed by the
      // allgather.
      send_idx = recv_idx;
      partner_mask >>= 1;
      slice_mask <<= 1;
      if (partner_mask > 0) {
        last_idx = recv_idx + pow2 / slice_mask;
      }
    }
    // Do a recursive-doubling allgather.
    slice_mask >>= 1;
    partner_mask = 1;
    while (partner_mask < static_cast<unsigned int>(pow2)) {
      // Compute the real rank.
      int adjusted_partner = rank ^ partner_mask;
      int partner = (adjusted_partner < pow2_remainder) ?
        adjusted_partner * 2 + 1 :
        adjusted_partner + pow2_remainder;
      // The send/recv ranges are computed similarly to above.
      size_t send_start, send_end, recv_start, recv_end;
      // The check is done on the adjusted partner rank.
      if (rank < adjusted_partner) {
        // Except on the first iteration, update last_idx.
        if (slice_mask != static_cast<unsigned int>(pow2) / 2) {
          last_idx += pow2 / (slice_mask*2);
        }
        recv_idx = send_idx + pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[recv_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[last_idx - 1];
      } else {
        recv_idx = send_idx - pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[last_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[send_idx - 1];
      }
      MPI_Sendrecv(recvbuf + send_start, send_end - send_start, type, partner, tag,
                   recvbuf + recv_start, recv_end - recv_start, type, partner, tag,
                   mpi_comm, MPI_STATUS_IGNORE);
      // Update for next iteration.
      if (rank > adjusted_partner) {  // Check on adjusted partner.
        send_idx = recv_idx;
      }
      partner_mask <<= 1;
      slice_mask >>= 1;
    }
  }
  // Send the excluded ranks their data in the non-power-of-2 case.
  if (orig_rank < 2 * pow2_remainder) {
    if (orig_rank % 2 == 0) {
      MPI_Recv(recvbuf, count, type, orig_rank + 1, tag, mpi_comm,
               MPI_STATUS_IGNORE);
    } else {
      MPI_Send(recvbuf, count, type, orig_rank - 1, tag, mpi_comm);
    }
  }
  release_memory(recv_to);
}

template <typename T>
class MPIRabenseifnerAlState : public MPIAlState<T> {
 public:
  MPIRabenseifnerAlState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AlRequest req_) :
    MPIAlState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {}
  bool setup() override {
    bool r = MPIAlState<T>::setup();
    if (!r) {
      // Check if we're in the non-power-of-2 case.
      while (pow2 <= this->nprocs) pow2 <<= 1;
      pow2 >>= 1;
      pow2_remainder = this->nprocs - pow2;
      adjusted_rank = this->rank;
      this->recv_to = nullptr;  // Will allocate later.
      // Adjust rank and start a send/recv for data if needed.
      if (this->rank < 2 * pow2_remainder) {
        if (this->rank % 2 == 0) {
          MPI_Isend(this->recvbuf, this->count, this->type, this->rank + 1,
                    this->tag, this->comm, &(this->send_recv_reqs[0]));
          adjusted_rank = -1;  // Don't participate.
        } else {
          // Need to receive entire buffer.
          this->recv_to = get_memory<T>(this->count);
          MPI_Irecv(this->recv_to, this->count, this->type, this->rank - 1,
                    this->tag, this->comm, &(this->send_recv_reqs[0]));
          adjusted_rank /= 2;
        }
      } else {
        setup_comm_done = true;  // No send/recv on this process.
        adjusted_rank -= pow2_remainder;
      }
      if (adjusted_rank != -1) {
        // Compute slices of data to be moved.
        const size_t size_per_rank = this->count / pow2;
        const size_t remainder = this->count % pow2;
        slice_lengths.resize(pow2, size_per_rank);
        // Add in the remainder as evenly as possible.
        for (size_t i = 0; i < remainder; ++i) {
          slice_lengths[i] += 1;
        }
        slice_ends.resize(pow2);
        std::partial_sum(slice_lengths.begin(), slice_lengths.end(),
                         slice_ends.begin());
        // Receive at most half the data.
        if (this->recv_to == nullptr) {
          this->recv_to = get_memory<T>(slice_ends[pow2 / 2]);
        }
        partner_mask = pow2 >> 1;
        last_idx = pow2;
      }
    }
    return r;
  }
  bool step() override {
    // Complete setup communication, if any.
    if (!setup_comm_done) {
      if (this->test_send_recv()) {
        setup_comm_done = true;
        if (this->rank % 2) {
          // Received data, need to reduce it.
          this->reduction_op(this->recv_to, this->recvbuf, this->count);
        }
      } else {
        return false;
      }
    }
    // Complete final communication in the non-power-of-2 case.
    if (final_comm_started) {
      return this->test_send_recv();
    }
    if (adjusted_rank == -1) {
      // Just need to wait for data.
      MPI_Irecv(this->recvbuf, this->count, this->type, this->rank + 1,
                this->tag, this->comm, &(this->send_recv_reqs[0]));
      final_comm_started = true;
      return false;
    }
    if (phase == 0) {
      if (rs_step()) {
        // Switch to allgather for next step.
        phase = 1;
        started = false;
        slice_mask >>= 1;
        partner_mask = 1;
      }
      return false;
    } else {
      if (ag_step()) {
        // Done, but in the non-power-of-2 case we need to send to our partner.
        if (this->rank < 2 * pow2_remainder) {
          MPI_Isend(this->recvbuf, this->count, this->type, this->rank - 1,
                    this->tag, this->comm, &(this->send_recv_reqs[0]));
          final_comm_started = true;
          return false;
        } else {
          return true;
        }
      } else {
        return false;
      }
    }
  }
 private:
  /** 0 == reduce-scatter, 1 == allgather. */
  int phase = 0;
  /** Whether communication has started. */
  bool started = false;
  /** Whether the send/recv from non-power-of-2 setup has completed. */
  bool setup_comm_done = false;
  /** Whether the final send/recv from the non-power-of-2 case has started. */
  bool final_comm_started = false;
  /** Nearest power-of-2 <= nprocs. */
  int pow2 = 1;
  /** Processes left over in the non-power-of-2 case. */
  int pow2_remainder = 0;
  /**
   * Process's adjusted rank for the non-power-of-2 case.
   * This is equal to rank if nprocs is a power of 2.
   */
  int adjusted_rank = -1;
  /** Mask for computing the partner. */
  unsigned int partner_mask;
  /** Mask for computing the data slices to send. */
  unsigned int slice_mask = 1;
  /** Starting index for sending. */
  int send_idx = 0;
  /** Sending index for receiving. */
  int recv_idx = 0;
  /** End of the right-most chunks sent. */
  int last_idx;
  /** Lengths of each slice of data to be sent. */
  std::vector<size_t> slice_lengths;
  /** End of each slice of data to be send. */
  std::vector<size_t> slice_ends;
  bool rs_step() {
    bool test = this->test_send_recv();
    if (started && test) {
      const int adjusted_old_partner = this->adjusted_rank ^ partner_mask;
      size_t old_recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      size_t old_recv_end;
      if (adjusted_rank < adjusted_old_partner) {
        old_recv_end = slice_ends[send_idx - 1];
      } else {
        old_recv_end = slice_ends[last_idx - 1];
      }
      this->reduction_op(this->recv_to, this->recvbuf + old_recv_start,
                         old_recv_end - old_recv_start);
      send_idx = recv_idx;
      partner_mask >>= 1;
      slice_mask <<= 1;
      if (partner_mask == 0) {
        return true;
      }
      // This isn't updated on the last iteration.
      last_idx = recv_idx + pow2 / slice_mask;
    }
    if (test) {
      const int adjusted_partner = adjusted_rank ^ partner_mask;
      const int partner = (adjusted_partner < pow2_remainder) ?
        adjusted_partner * 2 + 1 :
        adjusted_partner + pow2_remainder;
      // Compute the range of data to send/receive.
      size_t send_start, send_end, recv_start, recv_end;
      if (adjusted_rank < adjusted_partner) {
        send_idx = recv_idx + pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[last_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[send_idx - 1];
      } else {
        recv_idx = send_idx + pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[recv_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[last_idx - 1];
      }
      this->start_send_recv(
        this->recvbuf + send_start, send_end - send_start, partner,
        this->recv_to, recv_end - recv_start, partner);
      started = true;
    }
    return false;
  }
  bool ag_step() {
    bool test = this->test_send_recv();
    if (started && test) {
      // Update state.
      const int adjusted_old_partner = adjusted_rank ^ partner_mask;
      if (adjusted_rank > adjusted_old_partner) {
        send_idx = recv_idx;
      }
      partner_mask <<= 1;
      slice_mask >>= 1;
      if (partner_mask >= static_cast<unsigned int>(pow2)) {
        return true;
      }
    }
    if (test) {
      const int adjusted_partner = adjusted_rank ^ partner_mask;
      const int partner = (adjusted_partner < pow2_remainder) ?
        adjusted_partner * 2 + 1 :
        adjusted_partner + pow2_remainder;
      // The send/recv ranges are computed similarly to above.
      size_t send_start, send_end, recv_start, recv_end;
      if (adjusted_rank < adjusted_partner) {
        // Except on the first iteration, update last_idx.
        if (slice_mask != static_cast<unsigned int>(pow2) / 2) {
          last_idx += pow2 / (slice_mask*2);
        }
        recv_idx = send_idx + pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[recv_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[last_idx - 1];
      } else {
        recv_idx = send_idx - pow2 / (slice_mask*2);
        send_start = slice_ends[send_idx] - slice_lengths[send_idx];
        send_end = slice_ends[last_idx - 1];
        recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
        recv_end = slice_ends[send_idx - 1];
      }
      this->start_send_recv(
        this->recvbuf + send_start, send_end - send_start, partner,
        this->recvbuf + recv_start, recv_end - recv_start, partner);
      started = true;
    }
    return false;
  }
};

/** Non-blocking Rabenseifner allreduce. */
template <typename T>
void nb_rabenseifner_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                               ReductionOperator op, Communicator& comm,
                               AlRequest& req) {
  req = get_free_request();
  MPIRabenseifnerAlState<T>* state =
    new MPIRabenseifnerAlState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  if (state->setup()) {
    req = NULL_REQUEST;
    return;
  }
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

/**
 * Use a pairwise-exchange reduce-scatter and ring allgather to perform the
 * allreduce.
 */
template <typename T>
void pe_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, Communicator& comm,
                       const int tag = default_tag) {
  if (count == 0) return;  // Nothing to do.
  assert_count_fits_mpi(count);
  int rank = comm.rank();
  int nprocs = comm.size();
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  if (sendbuf != IN_PLACE<T>()) {
    // Copy our data into the receive buffer.
    std::copy_n(sendbuf, count, recvbuf);
  }
  if (nprocs == 1) return;  // Only needed to copy data.
  // Compute the slices of data to be moved.
  const size_t size_per_rank = count / nprocs;
  const size_t remainder = count % nprocs;
  std::vector<size_t> slice_lengths(nprocs, size_per_rank);
  // Add in the remainder as evenly as possible.
  for (size_t i = 0; i < remainder; ++i) {
    slice_lengths[i] += 1;
  }
  std::vector<size_t> slice_ends(nprocs);
  std::partial_sum(slice_lengths.begin(), slice_lengths.end(),
                   slice_ends.begin());
  // Temporary buffer for receiving.
  T* recv_to = get_memory<T>(slice_lengths[0]);
  MPI_Datatype type = TypeMap<T>();
  auto reduction_op = ReductionMap<T>(op);
  // Do a pairwise-exchange reduce-scatter.
  for (int step = 1; step < nprocs; ++step) {
    // Compute source and destination for this step.
    const int src = (rank - step + nprocs) % nprocs;
    const int dst = (rank + step) % nprocs;
    const T* to_send = recvbuf + (slice_ends[dst] - slice_lengths[dst]);
    // Note: We always receive to our local chunk.
    MPI_Sendrecv(to_send, slice_lengths[dst], type, dst, tag,
                 recv_to, slice_lengths[rank], type, src, tag,
                 mpi_comm, MPI_STATUS_IGNORE);
    reduction_op(recv_to, recvbuf + (slice_ends[rank] - slice_lengths[rank]),
                 slice_lengths[rank]);
  }
  // Ring allgather.
  // No temporary buffer is needed here: Receive directly into recvbuf.
  // Compute source (left) and destination (right) for the ring.
  const int src = (rank - 1 + nprocs) % nprocs;
  const int dst = (rank + 1) % nprocs;
  int send_idx = rank;  // We accumulated our slice only.
  for (int step = 0; step < nprocs - 1; ++step) {
    // This computes the location the data should lie based on the current step.
    const int recv_idx = (rank - step - 1 + nprocs) % nprocs;
    const T* to_send = recvbuf + (slice_ends[send_idx] - slice_lengths[send_idx]);
    MPI_Sendrecv(to_send, slice_lengths[send_idx], type, dst, tag,
                 recvbuf + (slice_ends[recv_idx] - slice_lengths[recv_idx]),
                 slice_lengths[recv_idx], type, src, tag,
                 mpi_comm, MPI_STATUS_IGNORE);
    send_idx = recv_idx;  // Forward the data received.
  }
  release_memory(recv_to);
}

}  // namespace mpi
}  // namespace internal

/**
 * Supported allreduce algorithms.
 * This is used for requesting a particular algorithm. Use automatic to let the
 * library select for you.
 */
enum class MPIAllreduceAlgorithm {
  automatic,
  mpi_passthrough,
  mpi_recursive_doubling,
  mpi_ring,
  mpi_rabenseifner,
  mpi_pe_ring,
  mpi_biring
};

/** Return a textual name for an MPI allreduce algorithm. */
inline std::string algorithm_name(MPIAllreduceAlgorithm algo) {
  switch (algo) {
  case MPIAllreduceAlgorithm::automatic:
    return "automatic";
  case MPIAllreduceAlgorithm::mpi_passthrough:
    return "MPI passthrough";
  case MPIAllreduceAlgorithm::mpi_recursive_doubling:
    return "MPI recursive doubling";
  case MPIAllreduceAlgorithm::mpi_ring:
    return "MPI ring";
  case MPIAllreduceAlgorithm::mpi_rabenseifner:
    return "MPI Rabenseifner";
  case MPIAllreduceAlgorithm::mpi_pe_ring:
    return "MPI PE/ring";
  case MPIAllreduceAlgorithm::mpi_biring:
    return "MPI biring";
  default:
    return "unknown";
  }
}

class MPIBackend {
 public:
  using allreduce_algo_type = MPIAllreduceAlgorithm;
  using comm_type = MPICommunicator;
  using req_type = internal::AlRequest;
  static constexpr std::nullptr_t null_req = nullptr;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type algo,
                        const int tag = internal::mpi::default_tag) {
    if (algo == MPIAllreduceAlgorithm::automatic) {
      // TODO: Better algorithm selection/performance model.
      // TODO: Make tuneable.
      if (count <= 1<<9) {
        algo = MPIAllreduceAlgorithm::mpi_recursive_doubling;
      } else {
        algo = MPIAllreduceAlgorithm::mpi_rabenseifner;
      }
    }
    switch (algo) {
      case MPIAllreduceAlgorithm::mpi_passthrough:
        internal::mpi::passthrough_allreduce(sendbuf, recvbuf, count, op, comm);
        break;
      case MPIAllreduceAlgorithm::mpi_recursive_doubling:
        internal::mpi::recursive_doubling_allreduce(
          sendbuf, recvbuf, count, op, comm, tag);
        break;
      case MPIAllreduceAlgorithm::mpi_ring:
        internal::mpi::ring_allreduce(sendbuf, recvbuf, count, op, comm, false, 1, tag);
        break;
      case MPIAllreduceAlgorithm::mpi_rabenseifner:
        internal::mpi::rabenseifner_allreduce(sendbuf, recvbuf, count, op, comm, tag);
        break;
      case MPIAllreduceAlgorithm::mpi_pe_ring:
        internal::mpi::pe_ring_allreduce(sendbuf, recvbuf, count, op, comm, tag);
        break;
      case MPIAllreduceAlgorithm::mpi_biring:
        internal::mpi::ring_allreduce(sendbuf, recvbuf, count, op, comm, true, 1, tag);
        break;
      default:
        throw_al_exception("Invalid algorithm for Allreduce");
    }
  }
  
  template <typename T>
  static void Allreduce(T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type algo) {
    Allreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingAllreduce(
      const T* sendbuf, T* recvbuf, size_t count,
      ReductionOperator op,
      comm_type& comm,
      req_type& req,
      allreduce_algo_type algo) {
    if (algo == MPIAllreduceAlgorithm::automatic) {
      // TODO: Better algorithm selection/performance model.
      // TODO: Make tuneable.
      if (count <= 1<<9) {
        algo = MPIAllreduceAlgorithm::mpi_recursive_doubling;
      } else {
        algo = MPIAllreduceAlgorithm::mpi_rabenseifner;
      }
    }
    switch (algo) {
      case MPIAllreduceAlgorithm::mpi_passthrough:
        internal::mpi::nb_passthrough_allreduce(sendbuf, recvbuf, count, op, comm,
                                                req);
        break;
      case MPIAllreduceAlgorithm::mpi_recursive_doubling:
        internal::mpi::nb_recursive_doubling_allreduce(
            sendbuf, recvbuf, count, op, comm, req);
        break;
      case MPIAllreduceAlgorithm::mpi_ring:
        internal::mpi::nb_ring_allreduce(sendbuf, recvbuf, count, op, comm, req);
        break;
      case MPIAllreduceAlgorithm::mpi_rabenseifner:
        internal::mpi::nb_rabenseifner_allreduce(sendbuf, recvbuf, count, op, comm,
                                                 req);
        break;
        /*case MPIAllreduceAlgorithm::mpi_pe_ring:
          internal::mpi::nb_pe_ring_allreduce(sendbuf, recvbuf, count, op, comm, req);
          break;*/
      default:
        throw_al_exception("Invalid algorithm for NonblockingAllreduce");
    }
  }

  template <typename T>
  static void NonblockingAllreduce(
      T* recvbuf, size_t count,
      ReductionOperator op, comm_type& comm,
      req_type& req,
      allreduce_algo_type algo) {
    NonblockingAllreduce(internal::IN_PLACE<T>(), recvbuf, count, op, comm,
                         req, algo);
  }
};

template <>
inline bool Test<MPIBackend>(typename MPIBackend::req_type& req) {
  internal::ProgressEngine* pe = internal::get_progress_engine();
  return pe->is_complete(req);
}

template <>
inline void Wait<MPIBackend>(typename MPIBackend::req_type& req) {
  internal::ProgressEngine* pe = internal::get_progress_engine();
  pe->wait_for_completion(req);
}

}  // namespace Al
