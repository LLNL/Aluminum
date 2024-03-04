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

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "aluminum/base.hpp"
#include "aluminum/internal.hpp"
#include "aluminum/progress.hpp"
#include "aluminum/state.hpp"

#include "aluminum/mpi/communicator.hpp"
#include "aluminum/mpi/utils.hpp"
#include "aluminum/mpi/allgather.hpp"
#include "aluminum/mpi/allgatherv.hpp"
#include "aluminum/mpi/allreduce.hpp"
#include "aluminum/mpi/alltoall.hpp"
#include "aluminum/mpi/alltoallv.hpp"
#include "aluminum/mpi/barrier.hpp"
#include "aluminum/mpi/bcast.hpp"
#include "aluminum/mpi/gather.hpp"
#include "aluminum/mpi/gatherv.hpp"
#include "aluminum/mpi/multisendrecv.hpp"
#include "aluminum/mpi/reduce.hpp"
#include "aluminum/mpi/reduce_scatter.hpp"
#include "aluminum/mpi/reduce_scatterv.hpp"
#include "aluminum/mpi/scatter.hpp"
#include "aluminum/mpi/scatterv.hpp"
#include "aluminum/mpi/pt2pt.hpp"
#include "mpi/multisendrecv.hpp"
#include "mpi/utils.hpp"

namespace Al {
namespace internal {
namespace mpi {

/** MPI initialization. */
void init(int& argc, char**& argv, MPI_Comm world_comm);
/** MPI finalization. */
void finalize();

}  // namespace mpi
}  // namespace internal

/**
 * Supported allreduce algorithms.
 * This is used for requesting a particular algorithm. Use automatic to let the
 * library select for you.
 *
 * Note: These are kept for backwards compatibility and are all
 * equivalent to an automatic algorithm that passes through to MPI.
 */
enum class MPIAllreduceAlgorithm {
  automatic,
  mpi_passthrough,
  mpi_recursive_doubling,
  mpi_ring,
  mpi_rabenseifner,
  mpi_biring
};
/** Supported algorithms for collectives. */
enum class MPICollectiveAlgorithm {
  automatic
};

/** Return a textual name for an MPI allreduce algorithm. */
inline std::string algorithm_name(MPIAllreduceAlgorithm algo) {
  switch (algo) {
  case MPIAllreduceAlgorithm::automatic:
  case MPIAllreduceAlgorithm::mpi_passthrough:
  case MPIAllreduceAlgorithm::mpi_recursive_doubling:
  case MPIAllreduceAlgorithm::mpi_ring:
  case MPIAllreduceAlgorithm::mpi_rabenseifner:
  case MPIAllreduceAlgorithm::mpi_biring:
    return "automatic";
  default:
    return "unknown";
  }
}

/** Return a textual name for a collective algorithm. */
inline std::string algorithm_name(MPICollectiveAlgorithm algo) {
  switch (algo) {
  case MPICollectiveAlgorithm::automatic:
    return "automatic";
  default:
    return "unknown";
  }
}

class MPIBackend {
 public:
  using allreduce_algo_type = MPIAllreduceAlgorithm;
  using allgather_algo_type = MPICollectiveAlgorithm;
  using allgatherv_algo_type = MPICollectiveAlgorithm;
  using alltoall_algo_type = MPICollectiveAlgorithm;
  using alltoallv_algo_type = MPICollectiveAlgorithm;
  using barrier_algo_type = MPICollectiveAlgorithm;
  using bcast_algo_type = MPICollectiveAlgorithm;
  using gather_algo_type = MPICollectiveAlgorithm;
  using gatherv_algo_type = MPICollectiveAlgorithm;
  using reduce_algo_type = MPICollectiveAlgorithm;
  using reduce_scatter_algo_type = MPICollectiveAlgorithm;
  using reduce_scatterv_algo_type = MPICollectiveAlgorithm;
  using scatter_algo_type = MPICollectiveAlgorithm;
  using scatterv_algo_type = MPICollectiveAlgorithm;
  using comm_type = internal::mpi::MPICommunicator;
  using req_type = internal::mpi::AlMPIReq;
  static constexpr std::nullptr_t null_req = nullptr;

  template <typename T>
  static void Allreduce(const T* sendbuf, T* recvbuf, size_t count,
                        ReductionOperator op, comm_type& comm,
                        allreduce_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
      case MPIAllreduceAlgorithm::automatic:
      case MPIAllreduceAlgorithm::mpi_passthrough:
      case MPIAllreduceAlgorithm::mpi_recursive_doubling:
      case MPIAllreduceAlgorithm::mpi_ring:
      case MPIAllreduceAlgorithm::mpi_rabenseifner:
      case MPIAllreduceAlgorithm::mpi_biring:
        handle_serialized(internal::mpi::passthrough_allreduce<T>,
                          internal::mpi::passthrough_nb_allreduce<T>,
                          sendbuf, recvbuf, count, op, comm);
        break;
      default:
        throw_al_exception("Invalid algorithm");
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
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
      case MPIAllreduceAlgorithm::automatic:
      case MPIAllreduceAlgorithm::mpi_passthrough:
      case MPIAllreduceAlgorithm::mpi_recursive_doubling:
      case MPIAllreduceAlgorithm::mpi_ring:
      case MPIAllreduceAlgorithm::mpi_rabenseifner:
      case MPIAllreduceAlgorithm::mpi_biring:
        internal::mpi::passthrough_nb_allreduce(sendbuf, recvbuf, count, op, comm,
                                                req);
        break;
      default:
        throw_al_exception("Invalid algorithm");
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

  template <typename T>
  static void Send(const T* sendbuf, size_t count, int dest, comm_type& comm) {
    internal::mpi::assert_count_fits_mpi(count);
    handle_serialized(internal::mpi::passthrough_send<T>,
                      internal::mpi::passthrough_nb_send<T>,
                      sendbuf, count, dest, comm);
  }

  template <typename T>
  static void NonblockingSend(const T* sendbuf, size_t count, int dest,
                              comm_type& comm, req_type& req) {
    internal::mpi::assert_count_fits_mpi(count);
    internal::mpi::passthrough_nb_send(sendbuf, count, dest, comm, req);
  }

  template <typename T>
  static void Recv(T* recvbuf, size_t count, int src, comm_type& comm) {
    internal::mpi::assert_count_fits_mpi(count);
    handle_serialized(internal::mpi::passthrough_recv<T>,
                      internal::mpi::passthrough_nb_recv<T>,
                      recvbuf, count, src, comm);
  }

  template <typename T>
  static void NonblockingRecv(T* recvbuf, size_t count, int src,
                              comm_type& comm, req_type& req) {
    internal::mpi::assert_count_fits_mpi(count);
    internal::mpi::passthrough_nb_recv(recvbuf, count, src, comm, req);
  }

  template <typename T>
  static void SendRecv(const T* sendbuf, size_t send_count, int dest,
                       T* recvbuf, size_t recv_count, int src, comm_type& comm) {
    internal::mpi::assert_count_fits_mpi(send_count);
    internal::mpi::assert_count_fits_mpi(recv_count);
    handle_serialized(internal::mpi::passthrough_sendrecv<T>,
                      internal::mpi::passthrough_nb_sendrecv<T>,
                      sendbuf, send_count, dest, recvbuf, recv_count, src, comm);
  }

  template <typename T>
  static void SendRecv(T* buf, size_t count, int dest, int src, comm_type& comm) {
    SendRecv(internal::IN_PLACE<T>(), count, dest, buf, count, src, comm);
  }

  template <typename T>
  static void NonblockingSendRecv(const T* sendbuf, size_t send_count, int dest,
                                  T* recvbuf, size_t recv_count, int src,
                                  comm_type& comm, req_type& req) {
    internal::mpi::assert_count_fits_mpi(send_count);
    internal::mpi::assert_count_fits_mpi(recv_count);
    internal::mpi::passthrough_nb_sendrecv(sendbuf, send_count, dest,
                                           recvbuf, recv_count, src,
                                           comm, req);
  }

  template <typename T>
  static void NonblockingSendRecv(T* buf, size_t count, int dest, int src,
                                  comm_type& comm, req_type& req) {
    NonblockingSendRecv(internal::IN_PLACE<T>(), count, dest, buf, count, src,
                        comm, req);
  }

  template <typename T>
  static void MultiSendRecv(std::vector<const T*> send_buffers,
                            std::vector<size_t> send_counts,
                            std::vector<int> dests,
                            std::vector<T*> recv_buffers,
                            std::vector<size_t> recv_counts,
                            std::vector<int> srcs, comm_type& comm) {
    for (size_t i = 0; i < send_counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(send_counts[i]);
    }
    for (size_t i = 0; i < recv_counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(recv_counts[i]);
    }
    handle_serialized(internal::mpi::passthrough_multisendrecv<T>,
                      internal::mpi::passthrough_nb_multisendrecv<T>,
                      send_buffers, send_counts, dests,
                      recv_buffers, recv_counts, srcs, comm);
  }

  template <typename T>
  static void MultiSendRecv(std::vector<T*> buffers,
                            std::vector<size_t> counts, std::vector<int> dests,
                            std::vector<int> srcs, comm_type& comm) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    handle_serialized(internal::mpi::passthrough_inplace_multisendrecv<T>,
                      internal::mpi::passthrough_nb_inplace_multisendrecv<T>,
                      buffers, counts, dests, srcs, comm);
  }

  template <typename T>
  static void NonblockingMultiSendRecv(std::vector<const T*> send_buffers,
                                       std::vector<size_t> send_counts,
                                       std::vector<int> dests,
                                       std::vector<T*> recv_buffers,
                                       std::vector<size_t> recv_counts,
                                       std::vector<int> srcs, comm_type& comm,
                                       req_type& req) {
    for (size_t i = 0; i < send_counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(send_counts[i]);
    }
    for (size_t i = 0; i < recv_counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(recv_counts[i]);
    }
    internal::mpi::passthrough_nb_multisendrecv(
        send_buffers, send_counts, dests, recv_buffers, recv_counts, srcs,
        comm, req);
  }

  template <typename T>
  static void NonblockingMultiSendRecv(std::vector<T*> buffers,
                                       std::vector<size_t> counts,
                                       std::vector<int> dests,
                                       std::vector<int> srcs, comm_type& comm,
                                       req_type& req) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    internal::mpi::passthrough_nb_inplace_multisendrecv(
      buffers, counts, dests, srcs, comm, req);
  }

  template <typename T>
  static void Allgather(
    const T* sendbuf, T* recvbuf, size_t count,
    comm_type& comm, allgather_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_allgather<T>,
                        internal::mpi::passthrough_nb_allgather<T>,
                        sendbuf, recvbuf, count, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Allgather(
    T* buffer, size_t count, comm_type& comm, allgather_algo_type algo) {
    Allgather(internal::IN_PLACE<T>(), buffer, count, comm, algo);
  }

  template <typename T>
  static void NonblockingAllgather(
    const T* sendbuf, T* recvbuf, size_t count,
    comm_type& comm, req_type& req, allgather_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_allgather(
        sendbuf, recvbuf, count, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingAllgather(
    T* buffer, size_t count,
    comm_type& comm, req_type& req, allgather_algo_type algo) {
    NonblockingAllgather(internal::IN_PLACE<T>(), buffer, count, comm, req,
                         algo);
  }

  template <typename T>
  static void Allgatherv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, std::vector<size_t> displs,
    comm_type& comm, allgatherv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_allgatherv<T>,
                        internal::mpi::passthrough_nb_allgatherv<T>,
                        sendbuf, recvbuf, counts, displs, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Allgatherv(
    T* buffer,
    std::vector<size_t> counts, std::vector<size_t> displs,
    comm_type& comm, allgatherv_algo_type algo) {
    Allgatherv(internal::IN_PLACE<T>(), buffer, counts, displs, comm, algo);
  }

  template <typename T>
  static void NonblockingAllgatherv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, std::vector<size_t> displs,
    comm_type& comm, req_type& req, allgatherv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_allgatherv(
        sendbuf, recvbuf, counts, displs, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingAllgatherv(
    T* buffer,
    std::vector<size_t> counts, std::vector<size_t> displs,
    comm_type& comm, req_type& req, allgatherv_algo_type algo) {
    NonblockingAllgatherv(internal::IN_PLACE<T>(), buffer, counts, displs, comm,
                          req, algo);
  }

  template <typename T>
  static void Alltoall(
    const T* sendbuf, T* recvbuf, size_t count,
    comm_type& comm, alltoall_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_alltoall<T>,
                        internal::mpi::passthrough_nb_alltoall<T>,
                        sendbuf, recvbuf, count, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Alltoall(
    T* buffer, size_t count, comm_type& comm, alltoall_algo_type algo) {
    Alltoall(internal::IN_PLACE<T>(), buffer, count, comm, algo);
  }

  template <typename T>
  static void NonblockingAlltoall(
    const T* sendbuf, T* recvbuf, size_t count,
    comm_type& comm, req_type& req, alltoall_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_alltoall(
        sendbuf, recvbuf, count, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingAlltoall(
    T* buffer, size_t count,
    comm_type& comm, req_type& req, alltoall_algo_type algo) {
    NonblockingAlltoall(internal::IN_PLACE<T>(), buffer, count, comm, req,
                        algo);
  }

  template <typename T>
  static void Alltoallv(
    const T* sendbuf,
    std::vector<size_t> send_counts, std::vector<size_t> send_displs,
    T* recvbuf,
    std::vector<size_t> recv_counts, std::vector<size_t> recv_displs,
    comm_type& comm, alltoallv_algo_type algo) {
    for (size_t i = 0; i < send_counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(send_counts[i]);
      internal::mpi::assert_count_fits_mpi(recv_counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_alltoallv<T>,
                        internal::mpi::passthrough_nb_alltoallv<T>,
                        sendbuf, send_counts, send_displs,
                        recvbuf, recv_counts, recv_displs, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Alltoallv(
    T* buffer, std::vector<size_t> counts, std::vector<size_t> displs,
    comm_type& comm, alltoallv_algo_type algo) {
    Alltoallv(internal::IN_PLACE<T>(), counts, displs, buffer, counts, displs,
              comm, algo);
  }

  template <typename T>
  static void NonblockingAlltoallv(
    const T* sendbuf,
    std::vector<size_t> send_counts, std::vector<size_t> send_displs,
    T* recvbuf,
    std::vector<size_t> recv_counts, std::vector<size_t> recv_displs,
    comm_type& comm, req_type& req, alltoallv_algo_type algo) {
    for (size_t i = 0; i < send_counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(send_counts[i]);
      internal::mpi::assert_count_fits_mpi(recv_counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_alltoallv(
        sendbuf, send_counts, send_displs,
        recvbuf, recv_counts, recv_displs,
        comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingAlltoallv(
    T* buffer, std::vector<size_t> counts, std::vector<size_t> displs,
    comm_type& comm, req_type& req, alltoallv_algo_type algo) {
    NonblockingAlltoallv(internal::IN_PLACE<T>(), counts, displs,
                         buffer, counts, displs, comm, req, algo);
  }

  static void Barrier(comm_type& comm, barrier_algo_type algo) {
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_barrier,
                        internal::mpi::passthrough_nb_barrier,
                        comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  static void NonblockingBarrier(comm_type& comm, req_type& req,
                                 barrier_algo_type algo) {
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_barrier(comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Bcast(T* buf, size_t count, int root, comm_type& comm,
                    bcast_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_bcast<T>,
                        internal::mpi::passthrough_nb_bcast<T>,
                        buf, count, root, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingBcast(
    T* buf, size_t count, int root,
    comm_type& comm, req_type& req, bcast_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_bcast(
        buf, count, root, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Gather(
    const T* sendbuf, T* recvbuf, size_t count, int root,
    comm_type& comm, gather_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_gather<T>,
                        internal::mpi::passthrough_nb_gather<T>,
                        sendbuf, recvbuf, count, root, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Gather(
    T* buffer, size_t count, int root, comm_type& comm, gather_algo_type algo) {
    Gather(internal::IN_PLACE<T>(), buffer, count, root, comm, algo);
  }

  template <typename T>
  static void NonblockingGather(
    const T* sendbuf, T* recvbuf, size_t count, int root,
    comm_type& comm, req_type& req, gather_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_gather(sendbuf, recvbuf, count, root, comm,
                                           req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingGather(
    T* buffer, size_t count, int root,
    comm_type& comm, req_type& req, gather_algo_type algo) {
    NonblockingGather(internal::IN_PLACE<T>(), buffer, count, root,
                      comm, req, algo);
  }

  template <typename T>
  static void Gatherv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, gatherv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_gatherv<T>,
                        internal::mpi::passthrough_nb_gatherv<T>,
                        sendbuf, recvbuf, counts, displs, root, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Gatherv(
    T* buffer,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, gatherv_algo_type algo) {
    Gatherv(internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, algo);
  }

  template <typename T>
  static void NonblockingGatherv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, req_type& req, gatherv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_gatherv(
        sendbuf, recvbuf, counts, displs, root, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingGatherv(
    T* buffer,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, req_type& req, gatherv_algo_type algo) {
    NonblockingGatherv(
      internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, req, algo);
  }

  template <typename T>
  static void Reduce(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op, int root,
    comm_type& comm, reduce_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_reduce<T>,
                        internal::mpi::passthrough_nb_reduce<T>,
                        sendbuf, recvbuf, count, op, root, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Reduce(
    T* buffer, size_t count, ReductionOperator op, int root, comm_type& comm,
    reduce_algo_type algo) {
    Reduce(internal::IN_PLACE<T>(), buffer, count, op, root, comm, algo);
  }

  template <typename T>
  static void NonblockingReduce(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op, int root,
    comm_type& comm, req_type& req, reduce_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_reduce(sendbuf, recvbuf, count, op, root,
                                           comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingReduce(
    T* buffer, size_t count, ReductionOperator op, int root,
    comm_type& comm, req_type& req, reduce_algo_type algo) {
    NonblockingReduce(internal::IN_PLACE<T>(), buffer, count, op, root, comm, req, algo);
  }

  template <typename T>
  static void Reduce_scatter(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
    comm_type& comm, reduce_scatter_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_reduce_scatter<T>,
                        internal::mpi::passthrough_nb_reduce_scatter<T>,
                        sendbuf, recvbuf, count, op, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Reduce_scatter(
    T* buffer, size_t count, ReductionOperator op,
    comm_type& comm, reduce_scatter_algo_type algo) {
    Reduce_scatter(internal::IN_PLACE<T>(), buffer, count, op, comm, algo);
  }

  template <typename T>
  static void NonblockingReduce_scatter(
    const T* sendbuf, T* recvbuf, size_t count, ReductionOperator op,
    comm_type& comm, req_type& req, reduce_scatter_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_reduce_scatter(sendbuf, recvbuf, count, op,
                                                   comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingReduce_scatter(
    T* buffer, size_t count, ReductionOperator op,
    comm_type& comm, req_type& req, reduce_scatter_algo_type algo) {
    NonblockingReduce_scatter(internal::IN_PLACE<T>(), buffer, count, op, comm,
                              req, algo);
  }

  template <typename T>
  static void Reduce_scatterv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, ReductionOperator op,
    comm_type& comm, reduce_scatterv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_reduce_scatterv<T>,
                        internal::mpi::passthrough_nb_reduce_scatterv<T>,
                        sendbuf, recvbuf, counts, op, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Reduce_scatterv(
    T* buffer, std::vector<size_t> counts, ReductionOperator op,
    comm_type& comm, reduce_scatterv_algo_type algo) {
    Reduce_scatterv(internal::IN_PLACE<T>(), buffer, counts, op,
                    comm, algo);
  }

  template <typename T>
  static void NonblockingReduce_scatterv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, ReductionOperator op,
    comm_type& comm, req_type& req, reduce_scatterv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_reduce_scatterv(
        sendbuf, recvbuf, counts, op, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingReduce_scatterv(
    T* buffer, std::vector<size_t> counts, ReductionOperator op,
    comm_type& comm, req_type& req, reduce_scatterv_algo_type algo) {
    NonblockingReduce_scatterv(
      internal::IN_PLACE<T>(), buffer, counts, op, comm, req, algo);
  }

  template <typename T>
  static void Scatter(
    const T* sendbuf, T* recvbuf, size_t count, int root,
    comm_type& comm, scatter_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_scatter<T>,
                        internal::mpi::passthrough_nb_scatter<T>,
                        sendbuf, recvbuf, count, root, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Scatter(
    T* buffer, size_t count, int root, comm_type& comm, scatter_algo_type algo) {
    Scatter(internal::IN_PLACE<T>(), buffer, count, root, comm, algo);
  }

  template <typename T>
  static void NonblockingScatter(
    const T* sendbuf, T* recvbuf, size_t count, int root,
    comm_type& comm, req_type& req, scatter_algo_type algo) {
    internal::mpi::assert_count_fits_mpi(count);
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_scatter(sendbuf, recvbuf, count, root,
                                            comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingScatter(
    T* buffer, size_t count, int root,
    comm_type& comm, req_type& req, scatter_algo_type algo) {
    NonblockingScatter(internal::IN_PLACE<T>(), buffer, count, root, comm, req,
                       algo);
  }

  template <typename T>
  static void Scatterv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, scatterv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      handle_serialized(internal::mpi::passthrough_scatterv<T>,
                        internal::mpi::passthrough_nb_scatterv<T>,
                        sendbuf, recvbuf, counts, displs, root, comm);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void Scatterv(
    T* buffer,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, scatterv_algo_type algo) {
    Scatterv(internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, algo);
  }

  template <typename T>
  static void NonblockingScatterv(
    const T* sendbuf, T* recvbuf,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, req_type& req, scatterv_algo_type algo) {
    for (size_t i = 0; i < counts.size(); ++i) {
      internal::mpi::assert_count_fits_mpi(counts[i]);
    }
    switch (algo) {
    case MPICollectiveAlgorithm::automatic:
      internal::mpi::passthrough_nb_scatterv(
        sendbuf, recvbuf, counts, displs, root, comm, req);
      break;
    default:
      throw_al_exception("Invalid algorithm");
    }
  }

  template <typename T>
  static void NonblockingScatterv(
    T* buffer,
    std::vector<size_t> counts, std::vector<size_t> displs, int root,
    comm_type& comm, req_type& req, scatterv_algo_type algo) {
    NonblockingScatterv(
      internal::IN_PLACE<T>(), buffer, counts, displs, root, comm, req, algo);
  }

  template <typename T>
  static void RegisterMemory(T*, size_t, comm_type&) {
    // Registration is not supported with MPI, but we don't want this
    // to fail to compile.
  }

  template <typename T>
  static void UnregisterMemory(T*, comm_type&) {
    // Registration is not supported with MPI.
  }

  static std::string Name() { return "MPIBackend"; }

private:
  /**
   * Handle AL_MPI_SERIALIZE support by dispatching a call to either
   * blocking_func (when not serialized) or nonblocking_func followed
   * immediately by a wait (when serialized). Arguments are passed
   * directly to the function.
   */
  template <typename BlockingFunc, typename NonblockingFunc, typename... Args>
  static void handle_serialized(BlockingFunc blocking_func,
                                NonblockingFunc nonblocking_func,
                                Args&&... args) {
#ifdef AL_MPI_SERIALIZE
    req_type req;
    nonblocking_func(std::forward<Args>(args)..., req);
    Al::Wait<MPIBackend>(req);
    (void) blocking_func;
#else
    blocking_func(std::forward<Args>(args)...);
    (void) nonblocking_func;
#endif
  }
};

// Forward declare:
template <typename Backend> bool Test(typename Backend::req_type&);
template <>
inline bool Test<MPIBackend>(typename MPIBackend::req_type& req) {
  if (req == MPIBackend::null_req) {
    return true;
  }
  if (req->load(std::memory_order_acquire)) {
    req = MPIBackend::null_req;
  }
  return false;
}

// Forward declare:
template <typename Backend> void Wait(typename Backend::req_type&);
template <>
inline void Wait<MPIBackend>(typename MPIBackend::req_type& req) {
  if (req == MPIBackend::null_req) {
    return;
  }
  // Spin until the request has completed.
  while (!req->load(std::memory_order_acquire)) {}
  req = MPIBackend::null_req;
}

}  // namespace Al
