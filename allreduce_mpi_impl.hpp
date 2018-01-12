#pragma once

#include <algorithm>
#include <vector>
#include <numeric>

namespace allreduces {
namespace internal {
namespace mpi {

/** Used to map types to the associated MPI datatype. */
template <typename T>
inline MPI_Datatype TypeMap();
template <> inline MPI_Datatype TypeMap<int>() { return MPI_INT; }
template <> inline MPI_Datatype TypeMap<float>() { return MPI_FLOAT; }
template <> inline MPI_Datatype TypeMap<double>() { return MPI_DOUBLE; }

/** True if count elements can be sent by MPI. */
inline bool check_count_fits_mpi(size_t count) {
  return count <= static_cast<size_t>(std::numeric_limits<int>::max());
}
/** Throw an exception if count elements cannot be sent by MPI. */
inline void assert_count_fits_mpi(size_t count) {
  if (!check_count_fits_mpi(count)) {
    throw_allreduce_exception("Message count too large for MPI");
  }
}

/** Basic sum reduction. */
template <typename T>
void sum_reduction(const T* src, T* dest, size_t count) {
  if (count >= ALLREDUCE_MPI_MULTITHREAD_SUM_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] += src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] += src[i];
    }
  }
}
/** Basic prod reduction. */
template <typename T>
void prod_reduction(const T* src, T* dest, size_t count) {
  if (count >= ALLREDUCE_MPI_MULTITHREAD_PROD_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] *= src[i];
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] *= src[i];
    }
  }
}
/** Basic min reduction. */
template <typename T>
void min_reduction(const T* src, T* dest, size_t count) {
  if (count >= ALLREDUCE_MPI_MULTITHREAD_MINMAX_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::min(dest[i], src[i]);
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::min(dest[i], src[i]);
    }
  }
}
/** Basic max reduction. */
template <typename T>
void max_reduction(const T* src, T* dest, size_t count) {
  if (count >= ALLREDUCE_MPI_MULTITHREAD_MINMAX_THRESH) {
    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::max(dest[i], src[i]);
    }
  } else {
    for (size_t i = 0; i < count; ++i) {
      dest[i] = std::max(dest[i], src[i]);
    }
  }
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
  default:
    throw_allreduce_exception("Reduction operator not supported");
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
  default:
    throw_allreduce_exception("Reduction operator not supported");
  }
}

/** Base state class for MPI allreduces. */
template <typename T>
class MPIAllreduceState : public AllreduceState {
 public:
  MPIAllreduceState(const T* sendbuf_, T* recvbuf_, size_t count_,
                    ReductionOperator op_, Communicator& comm_,
                    AllreduceRequest req_) :
    AllreduceState(req_), sendbuf(sendbuf_), recvbuf(recvbuf_), recv_to(nullptr),
    count(count_) {
    comm = dynamic_cast<MPICommunicator&>(comm_).get_comm();
    type = TypeMap<T>();
    reduction_op = ReductionMap<T>(op_);
    rank = comm_.rank();
    nprocs = comm_.size();
    tag = dynamic_cast<MPICommunicator&>(comm_).get_free_tag();
  }
  ~MPIAllreduceState() override {
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

  /** Start a send/recv. */
  void start_send_recv(
    const void* send, int send_count, int dest,
    void* recv, int recv_count, int source) {
    MPI_Irecv(recv, recv_count, type, source, tag, comm, &(send_recv_reqs[0]));
    MPI_Isend(send, send_count, type, dest, tag, comm, &(send_recv_reqs[1]));
  }
  /** Return true if the outstanding send/recv has completed. */
  bool test_send_recv() {
    int flag;
    MPI_Testall(2, send_recv_reqs, &flag, MPI_STATUSES_IGNORE);
    return flag;
  }
};

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
class MPIPassthroughAllreduceState : public MPIAllreduceState<T> {
 public:
  MPIPassthroughAllreduceState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AllreduceRequest req_) :
    MPIAllreduceState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {
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

template <typename T>
void nb_passthrough_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                              ReductionOperator op, Communicator& comm,
                              AllreduceRequest& req) {
  req = get_free_request();
  MPIPassthroughAllreduceState<T>* state =
    new MPIPassthroughAllreduceState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  state->setup();
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

template <typename T>
void recursive_doubling_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                  ReductionOperator op, Communicator& comm) {
  if (count == 0) return;  // Nothing to do.
  assert_count_fits_mpi(count);
  int rank = comm.rank();
  int nprocs = comm.size();
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  // Currently only supports a power-of-2 number of processes.
  // TODO: Support any number of processors.
  if (nprocs & (nprocs - 1)) {
    throw_allreduce_exception("Recursive doubling requires a power-of-2 number"
                              " of processes");
  }
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
  while (mask < static_cast<unsigned int>(nprocs)) {
    int partner = rank ^ mask;
    MPI_Sendrecv(recvbuf, count, type, partner, 0,
                 recv_to, count, type, partner, 0,
                 mpi_comm, MPI_STATUS_IGNORE);
    reduction_op(recv_to, recvbuf, count);
    mask <<= 1;
  }
  release_memory(recv_to);
}

template <typename T>
class MPIRecursiveDoublingAllreduceState : public MPIAllreduceState<T> {
 public:
  MPIRecursiveDoublingAllreduceState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AllreduceRequest req_) :
    MPIAllreduceState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {}
  bool setup() override {
    bool r = MPIAllreduceState<T>::setup();
    if (!r) {
      this->recv_to = get_memory<T>(this->count);
    }
    return r;
  }
  bool step() override {
    const int partner = this->rank ^ mask;
    MPI_Sendrecv(this->recvbuf, this->count, this->type, partner, this->tag,
                 this->recv_to, this->count, this->type, partner, this->tag,
                 this->comm, MPI_STATUS_IGNORE);
    this->reduction_op(this->recv_to, this->recvbuf, this->count);
    mask <<= 1;
    if (mask >= static_cast<unsigned int>(this->nprocs)) {
      return true;  // Done.
    }
    return false;
  }
 private:
  unsigned int mask = 1;
};

template <typename T>
void nb_recursive_doubling_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                                     ReductionOperator op, Communicator& comm,
                                     AllreduceRequest& req) {
  if (comm.size() & (comm.size() - 1)) {
    throw_allreduce_exception("Recursive doubling requires a power-of-2 number"
                              " of processes");
  }
  req = get_free_request();
  MPIRecursiveDoublingAllreduceState<T>* state =
    new MPIRecursiveDoublingAllreduceState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  if (state->setup()) {
    req = NULL_REQUEST;
    return;
  }
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

template <typename T>
void ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                    ReductionOperator op, Communicator& comm) {
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
  // Compute source (left) and destination (right) for the ring.
  const int src = (rank - 1 + nprocs) % nprocs;
  const int dst = (rank + 1) % nprocs;
  MPI_Datatype type = TypeMap<T>();
  auto reduction_op = ReductionMap<T>(op);
  // Ring reduce-scatter.
  for (int step = 0; step < nprocs - 1; ++step) {
    // Determine which slices we're sending and receiving.
    const int send_idx = (rank - step + nprocs) % nprocs;
    const int recv_idx = (rank - step - 1 + nprocs) % nprocs;
    const T* to_send = recvbuf + (slice_ends[send_idx] - slice_lengths[send_idx]);
    MPI_Sendrecv(to_send, slice_lengths[send_idx], type, dst, 0,
                 recv_to, slice_lengths[recv_idx], type, src, 0,
                 mpi_comm, MPI_STATUS_IGNORE);
    reduction_op(recv_to, recvbuf + (slice_ends[recv_idx] - slice_lengths[recv_idx]),
                 slice_lengths[recv_idx]);
  }
  // Ring allgather.
  // No temporary buffer is needed here: Receive directly into recvbuf.
  // Compute the initial slice we send as the final slice we received.
  int send_idx = (rank + 1) % nprocs;
  for (int step = 0; step < nprocs - 1; ++step) {
    const int recv_idx = (rank - step + nprocs) % nprocs;
    const T* to_send = recvbuf + (slice_ends[send_idx] - slice_lengths[send_idx]);
    MPI_Sendrecv(to_send, slice_lengths[send_idx], type, dst, 0,
                 recvbuf + (slice_ends[recv_idx] - slice_lengths[recv_idx]),
                 slice_lengths[recv_idx], type, src, 0,
                 mpi_comm, MPI_STATUS_IGNORE);
    send_idx = recv_idx;  // Forward the data received.
  }
  release_memory(recv_to);
}

template <typename T>
class MPIRingAllreduceState : public MPIAllreduceState<T> {
 public:
  MPIRingAllreduceState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AllreduceRequest req_) :
    MPIAllreduceState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {}
  bool setup() override {
    bool r = MPIAllreduceState<T>::setup();
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
    const int send_idx = (this->rank - cur_step + this->nprocs) % this->nprocs;
    const int recv_idx = (this->rank - cur_step - 1 + this->nprocs) % this->nprocs;
    const T* to_send = this->recvbuf +
      (slice_ends[send_idx] - slice_lengths[send_idx]);
    MPI_Sendrecv(to_send, slice_lengths[send_idx], this->type, dst, this->tag,
                 this->recv_to, slice_lengths[recv_idx], this->type, src,
                 this->tag, this->comm, MPI_STATUS_IGNORE);
    this->reduction_op(
      this->recv_to,
      this->recvbuf + (slice_ends[recv_idx] - slice_lengths[recv_idx]),
      slice_lengths[recv_idx]);
    ++cur_step;
    if (cur_step >= this->nprocs - 1) {
      return true;  // Done.
    }
    return false;
  }
  bool ag_step() {
    const int recv_idx = (this->rank - cur_step + this->nprocs) % this->nprocs;
    const T* to_send =
      this->recvbuf + (slice_ends[ag_send_idx] - slice_lengths[ag_send_idx]);
    MPI_Sendrecv(to_send, slice_lengths[ag_send_idx], this->type, dst,
                 this->tag, this->recvbuf +
                 (slice_ends[recv_idx] - slice_lengths[recv_idx]),
                 slice_lengths[recv_idx], this->type, src, this->tag,
                 this->comm, MPI_STATUS_IGNORE);
    ag_send_idx = recv_idx;  // Forward data received.
    ++cur_step;
    if (cur_step >= this->nprocs - 1) {
      return true;  // Done.
    }
    return false;
  }
};

template <typename T>
void nb_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, Communicator& comm,
                       AllreduceRequest& req) {
  req = get_free_request();
  MPIRingAllreduceState<T>* state =
    new MPIRingAllreduceState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  if (state->setup()) {
    req = NULL_REQUEST;
    return;
  }
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

template <typename T>
void rabenseifner_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                            ReductionOperator op, Communicator& comm) {
  if (count == 0) return;  // Nothing to do.
  assert_count_fits_mpi(count);
  int rank = comm.rank();
  int nprocs = comm.size();
  MPI_Comm mpi_comm = dynamic_cast<MPICommunicator&>(comm).get_comm();
  // Currently only supports a power-of-2 number of processes.
  // TODO: Support any number of processors.
  if (nprocs & (nprocs - 1)) {
    throw_allreduce_exception("Rabenseifner requires a power-of-2 number"
                              " of processes");
  }
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
  // We receive at most half the data.
  T* recv_to = get_memory<T>(slice_ends[nprocs / 2]);
  MPI_Datatype type = TypeMap<T>();
  auto reduction_op = ReductionMap<T>(op);
  // Do a recursive-halving reduce-scatter.
  unsigned int partner_mask = nprocs >> 1;
  // Used to compute the number of slices sent in a step.
  unsigned int slice_mask = 1;
  int send_idx = 0;  // Starting index for sending.
  int recv_idx = 0;  // Starting index for receiving.
  int last_idx = nprocs;  // End of right-most region.
  while (partner_mask > 0) {
    int partner = rank ^ partner_mask;
    // Compute the range of data to send and receive.
    size_t send_start, send_end, recv_start, recv_end;
    if (rank < partner) {
      send_idx = recv_idx + nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[last_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[send_idx - 1];
    } else {
      recv_idx = send_idx + nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[recv_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[last_idx - 1];
    }
    MPI_Sendrecv(recvbuf + send_start, send_end - send_start, type, partner, 0,
                 recv_to, recv_end - recv_start, type, partner, 0,
                 mpi_comm, MPI_STATUS_IGNORE);
    reduction_op(recv_to, recvbuf + recv_start, recv_end - recv_start);
    // Update for the next iteration, except last_idx, which is needed by the
    // allgather.
    send_idx = recv_idx;
    partner_mask >>= 1;
    slice_mask <<= 1;
    if (partner_mask > 0) {
      last_idx = recv_idx + nprocs / slice_mask;
    }
  }
  // Do a recursive-doubling allgather.
  slice_mask >>= 1;
  partner_mask = 1;
  while (partner_mask < static_cast<unsigned int>(nprocs)) {
    int partner = rank ^ partner_mask;
    // The send/recv ranges are computed similarly to above.
    size_t send_start, send_end, recv_start, recv_end;
    if (rank < partner) {
      // Except on the first iteration, update last_idx.
      if (slice_mask != static_cast<unsigned int>(nprocs) / 2) {
        last_idx += nprocs / (slice_mask*2);
      }
      recv_idx = send_idx + nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[recv_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[last_idx - 1];
    } else {
      recv_idx = send_idx - nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[last_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[send_idx - 1];
    }
    MPI_Sendrecv(recvbuf + send_start, send_end - send_start, type, partner, 0,
                 recvbuf + recv_start, recv_end - recv_start, type, partner, 0,
                 mpi_comm, MPI_STATUS_IGNORE);
    // Update for next iteration.
    if (rank > partner) {
      send_idx = recv_idx;
    }
    partner_mask <<= 1;
    slice_mask >>= 1;
  }
  release_memory(recv_to);
}

template <typename T>
class MPIRabenseifnerAllreduceState : public MPIAllreduceState<T> {
 public:
  MPIRabenseifnerAllreduceState(
    const T* sendbuf_, T* recvbuf_, size_t count_,
    ReductionOperator op_, Communicator& comm_, AllreduceRequest req_) :
    MPIAllreduceState<T>(sendbuf_, recvbuf_, count_, op_, comm_, req_) {}
  bool setup() override {
    bool r = MPIAllreduceState<T>::setup();
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
      // Receive at most half the data.
      this->recv_to = get_memory<T>(slice_ends[this->nprocs / 2]);
      partner_mask = this->nprocs >> 1;
      last_idx = this->nprocs;
    }
    return r;
  }
  bool step() override {
    if (phase == 0) {
      if (rs_step()) {
        // Switch to allgather for next step.
        phase = 1;
        slice_mask >>= 1;
        partner_mask = 1;
      }
      return false;
    } else {
      return ag_step();
    }
  }
 private:
  /** 0 == reduce-scatter, 1 == allgather. */
  int phase = 0;
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
    const int partner = this->rank ^ partner_mask;
    // Compute the range of data to send/receive.
    size_t send_start, send_end, recv_start, recv_end;
    if (this->rank < partner) {
      send_idx = recv_idx + this->nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[last_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[send_idx - 1];
    } else {
      recv_idx = send_idx + this->nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[recv_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[last_idx - 1];
    }
    MPI_Sendrecv(this->recvbuf + send_start, send_end - send_start, this->type,
                 partner, this->tag, this->recv_to, recv_end - recv_start,
                 this->type, partner, this->tag, this->comm,
                 MPI_STATUS_IGNORE);
    this->reduction_op(this->recv_to, this->recvbuf + recv_start,
                       recv_end - recv_start);
    send_idx = recv_idx;
    partner_mask >>= 1;
    slice_mask <<= 1;
    if (partner_mask == 0) {
      return true;  // Done.
    }
    // This isn't updated on the last iteration.
    last_idx = recv_idx + this->nprocs / slice_mask;
    return false;
  }
  bool ag_step() {
    const int partner = this->rank ^ partner_mask;
    // The send/recv ranges are computed similarly to above.
    size_t send_start, send_end, recv_start, recv_end;
    if (this->rank < partner) {
      // Except on the first iteration, update last_idx.
      if (slice_mask != static_cast<unsigned int>(this->nprocs) / 2) {
        last_idx += this->nprocs / (slice_mask*2);
      }
      recv_idx = send_idx + this->nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[recv_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[last_idx - 1];
    } else {
      recv_idx = send_idx - this->nprocs / (slice_mask*2);
      send_start = slice_ends[send_idx] - slice_lengths[send_idx];
      send_end = slice_ends[last_idx - 1];
      recv_start = slice_ends[recv_idx] - slice_lengths[recv_idx];
      recv_end = slice_ends[send_idx - 1];
    }
    MPI_Sendrecv(this->recvbuf + send_start, send_end - send_start,
                 this->type, partner, this->tag,
                 this->recvbuf + recv_start, recv_end - recv_start,
                 this->type, partner, this->tag,
                 this->comm, MPI_STATUS_IGNORE);
    if (this->rank > partner) {
      send_idx = recv_idx;
    }
    partner_mask <<= 1;
    slice_mask >>= 1;
    if (partner_mask >= static_cast<unsigned int>(this->nprocs)) {
      return true;  // Done.
    }
    return false;
  }
};

template <typename T>
void nb_rabenseifner_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                               ReductionOperator op, Communicator& comm,
                               AllreduceRequest& req) {
  if (comm.size() & (comm.size() - 1)) {
    throw_allreduce_exception("Rabenseifner doubling requires a power-of-2 number"
                              " of processes");
  }
  req = get_free_request();
  MPIRabenseifnerAllreduceState<T>* state =
    new MPIRabenseifnerAllreduceState<T>(
      sendbuf, recvbuf, count, op, comm, req);
  if (state->setup()) {
    req = NULL_REQUEST;
    return;
  }
  ProgressEngine* pe = get_progress_engine();
  pe->enqueue(state);
}

template <typename T>
void pe_ring_allreduce(const T* sendbuf, T* recvbuf, size_t count,
                       ReductionOperator op, Communicator& comm) {
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
    MPI_Sendrecv(to_send, slice_lengths[dst], type, dst, 0,
                 recv_to, slice_lengths[rank], type, src, 0,
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
    MPI_Sendrecv(to_send, slice_lengths[send_idx], type, dst, 0,
                 recvbuf + (slice_ends[recv_idx] - slice_lengths[recv_idx]),
                 slice_lengths[recv_idx], type, src, 0,
                 mpi_comm, MPI_STATUS_IGNORE);
    send_idx = recv_idx;  // Forward the data received.
  }
  release_memory(recv_to);
}

}  // namespace mpi
}  // namespace internal
}  // namespace allreduces
