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

#include "Al.hpp"

template <typename Backend, typename T>
class OpRunner<AlOperation::allgather, Backend, T> :
  public OpRunnerShim<AlOperation::allgather, Backend, T,
                      OpRunner<AlOperation::allgather, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::allgather, Backend, T>;
  using OpRunnerShim<AlOperation::allgather, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::allgather,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    size_t size = this->get_options().inplace
      ? output.size() / comm.size()
      : input.size();
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.allgather_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Allgather<Backend>(input.data(), output.data(), size, comm, algo); },
      [&]() { Al::Allgather<Backend>(output.data(), size, comm, algo); },
      [&]() { Al::NonblockingAllgather<Backend>(input.data(), output.data(), size, comm, req, algo); },
      [&]() { Al::NonblockingAllgather<Backend>(output.data(), size, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Allgather(this->buf_or_inplace(input.data()), input.size(),
                  Al::internal::mpi::TypeMap<T>(),
                  output.data(),
                  this->get_options().inplace ? output.size() / comm.size() : input.size(),
                  Al::internal::mpi::TypeMap<T>(),
                  comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& /*comm*/) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return base_size;
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& comm) {
    return base_size * comm.size();
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::allgatherv, Backend, T> :
  public OpRunnerShim<AlOperation::allgatherv, Backend, T,
                      OpRunner<AlOperation::allgatherv, Backend, T>> {
public:
  using OpRunnerShim<AlOperation::allgatherv, Backend, T,
                     OpRunner<AlOperation::allgatherv, Backend, T>>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::allgatherv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    auto send_counts = this->get_options().send_counts;
    auto send_displs = this->get_options().send_displs;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.allgatherv_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Allgatherv<Backend>(input.data(), output.data(), send_counts, send_displs, comm, algo); },
      [&]() { Al::Allgatherv<Backend>(output.data(), send_counts, send_displs, comm, algo); },
      [&]() { Al::NonblockingAllgatherv<Backend>(input.data(), output.data(), send_counts, send_displs, comm, req, algo); },
      [&]() { Al::NonblockingAllgatherv<Backend>(output.data(), send_counts, send_displs, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    std::vector<int> counts = Al::internal::mpi::intify_size_t_vector(
      this->get_options().send_counts);
    std::vector<int> displs = Al::internal::mpi::intify_size_t_vector(
      this->get_options().send_displs);
    MPI_Allgatherv(this->buf_or_inplace(input.data()), counts[comm.rank()],
                   Al::internal::mpi::TypeMap<T>(),
                   output.data(), counts.data(), displs.data(),
                   Al::internal::mpi::TypeMap<T>(), comm.get_comm());
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return this->get_options().send_counts[comm.rank()];
    }
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& /*comm*/) {
    return std::accumulate(this->get_options().send_counts.begin(),
                           this->get_options().send_counts.end(), size_t{0});
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::allreduce, Backend, T> :
  public OpRunnerShim<AlOperation::allreduce, Backend, T,
                      OpRunner<AlOperation::allreduce, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::allreduce, Backend, T>;
  using OpRunnerShim<AlOperation::allreduce, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::allreduce,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    auto reduction_op = this->get_options().reduction_op;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.allreduce_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Allreduce<Backend>(input.data(), output.data(), input.size(), reduction_op, comm, algo); },
      [&]() { Al::Allreduce<Backend>(output.data(), output.size(), reduction_op, comm, algo); },
      [&]() { Al::NonblockingAllreduce<Backend>(input.data(), output.data(), input.size(), reduction_op, comm, req, algo); },
      [&]() { Al::NonblockingAllreduce<Backend>(output.data(), output.size(), reduction_op, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op(
      this->get_options().reduction_op);
    MPI_Allreduce(this->buf_or_inplace(input.data()), output.data(),
                  this->get_options().inplace ? output.size() : input.size(),
                  Al::internal::mpi::TypeMap<T>(),
                  reduction_op, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& /*comm*/) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return base_size;
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& /*comm*/) {
    return base_size;
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::alltoall, Backend, T> :
  public OpRunnerShim<AlOperation::alltoall, Backend, T,
                      OpRunner<AlOperation::alltoall, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::alltoall, Backend, T>;
  using OpRunnerShim<AlOperation::alltoall, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::alltoall,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    size_t size = this->get_options().inplace
      ? output.size() / comm.size()
      : input.size() / comm.size();
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.alltoall_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Alltoall<Backend>(input.data(), output.data(), size, comm, algo); },
      [&]() { Al::Alltoall<Backend>(output.data(), size, comm, algo); },
      [&]() { Al::NonblockingAlltoall<Backend>(input.data(), output.data(), size, comm, req, algo); },
      [&]() { Al::NonblockingAlltoall<Backend>(output.data(), size, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    size_t size = this->get_options().inplace
      ? output.size() / comm.size()
      : input.size() / comm.size();
    MPI_Alltoall(this->buf_or_inplace(input.data()), size,
                 Al::internal::mpi::TypeMap<T>(),
                 output.data(), size,
                 Al::internal::mpi::TypeMap<T>(),
                 comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return base_size * comm.size();
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& comm) {
    return base_size * comm.size();
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::alltoallv, Backend, T> :
  public OpRunnerShim<AlOperation::alltoallv, Backend, T,
                      OpRunner<AlOperation::alltoallv, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::alltoallv, Backend, T>;
  using OpRunnerShim<AlOperation::alltoallv, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::alltoallv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    auto send_counts = this->get_options().send_counts;
    auto recv_counts = this->get_options().recv_counts;
    auto send_displs = this->get_options().send_displs;
    auto recv_displs = this->get_options().recv_displs;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.alltoallv_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Alltoallv<Backend>(input.data(), send_counts, send_displs, output.data(), recv_counts, recv_displs, comm, algo); },
      [&]() { Al::Alltoallv<Backend>(output.data(), send_counts, send_displs, comm, algo); },
      [&]() { Al::NonblockingAlltoallv<Backend>(input.data(), send_counts, send_displs, output.data(), recv_counts, recv_displs, comm, req, algo); },
      [&]() { Al::NonblockingAlltoallv<Backend>(output.data(), send_counts, send_displs, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    std::vector<int> send_counts =
        Al::internal::mpi::intify_size_t_vector(this->get_options().send_counts);
    std::vector<int> send_displs =
        Al::internal::mpi::intify_size_t_vector(this->get_options().send_displs);
    std::vector<int> recv_counts =
        Al::internal::mpi::intify_size_t_vector(this->get_options().recv_counts);
    std::vector<int> recv_displs =
        Al::internal::mpi::intify_size_t_vector(this->get_options().recv_displs);
    MPI_Alltoallv(this->buf_or_inplace(input.data()),
                  send_counts.data(), send_displs.data(),
                  Al::internal::mpi::TypeMap<T>(),
                  output.data(),
                  recv_counts.data(), recv_displs.data(),
                  Al::internal::mpi::TypeMap<T>(),
                  comm.get_comm());
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& /*comm*/) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return std::accumulate(this->get_options().send_counts.begin(),
                             this->get_options().send_counts.end(), size_t{0});
    }
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& /*comm*/) {
    return std::accumulate(this->get_options().send_counts.begin(),
                             this->get_options().send_counts.end(), size_t{0});
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::barrier, Backend, T> :
  public OpRunnerShim<AlOperation::barrier, Backend, T,
                      OpRunner<AlOperation::barrier, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::barrier, Backend, T>;
  using OpRunnerShim<AlOperation::barrier, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::barrier,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& /*input*/,
                typename VectorType<T, Backend>::type& /*output*/,
                typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      std::cerr << "No in-place for barrier" << std::endl;
      std::abort();
    }
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.allgather_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Barrier<Backend>(comm, algo); },
      [&]() {},
      [&]() { Al::NonblockingBarrier<Backend>(comm, req, algo); },
      [&]() {});
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& /*input*/,
                    std::vector<T>& /*output*/,
                    typename Backend::comm_type& comm) {
    MPI_Barrier(comm.get_comm());
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& /*comm*/) {
    return 0;
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& /*comm*/) {
    return 0;
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::bcast, Backend, T> :
  public OpRunnerShim<AlOperation::bcast, Backend, T,
                      OpRunner<AlOperation::bcast, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::bcast, Backend, T>;
  using OpRunnerShim<AlOperation::bcast, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::bcast,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& /*input*/,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    if (!this->get_options().inplace) {
      std::cerr << "Bcast must be in-place" << std::endl;
      std::abort();
    }
    int root = this->get_options().root;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.allgather_algo;
    this->inplace_nb_dispatch(
      [&]() {},
      [&]() { Al::Bcast<Backend>(output.data(), output.size(), root, comm, algo); },
      [&]() {},
      [&]() { Al::NonblockingBcast<Backend>(output.data(), output.size(), root, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& /*input*/,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Bcast(output.data(), output.size(),
              Al::internal::mpi::TypeMap<T>(),
              this->get_options().root, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& comm) {
    if (comm.rank() == this->get_options().root) {
      return base_size;
    } else {
      return 0;
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& /*comm*/) {
    return base_size;
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::gather, Backend, T> :
  public OpRunnerShim<AlOperation::gather, Backend, T,
                      OpRunner<AlOperation::gather, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::gather, Backend, T>;
  using OpRunnerShim<AlOperation::gather, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::gather,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    size_t size = (comm.rank() == root)
      ? (this->get_options().inplace ? output.size() / comm.size() : input.size())
      : (this->get_options().inplace ? output.size() : input.size());
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.gather_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Gather<Backend>(input.data(), output.data(), size, root, comm, algo); },
      [&]() { Al::Gather<Backend>(output.data(), size, root, comm, algo); },
      [&]() { Al::NonblockingGather<Backend>(input.data(), output.data(), size, root, comm, req, algo); },
      [&]() { Al::NonblockingGather<Backend>(output.data(), size, root, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    size_t size = (comm.rank() == root)
      ? (this->get_options().inplace ? output.size() / comm.size() : input.size())
      : (this->get_options().inplace ? output.size() : input.size());
    // Account for in-place only being used at the root.
    void* sendbuf = (comm.rank() == root)
      ? this->buf_or_inplace(input.data())
      : (this->get_options().inplace ? output.data() : input.data());
    MPI_Gather(sendbuf, size,
               Al::internal::mpi::TypeMap<T>(),
               output.data(), size,
               Al::internal::mpi::TypeMap<T>(),
               root, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& /*comm*/) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return base_size;
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& comm) {
    if (comm.rank() == this->get_options().root) {
      return base_size * comm.size();
    } else {
      if (this->get_options().inplace) {
        return base_size;
      } else {
        return 0;
      }
    }
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::gatherv, Backend, T> :
  public OpRunnerShim<AlOperation::gatherv, Backend, T,
                      OpRunner<AlOperation::gatherv, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::gatherv, Backend, T>;
  using OpRunnerShim<AlOperation::gatherv, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::gatherv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    auto send_counts = this->get_options().send_counts;
    auto send_displs = this->get_options().send_displs;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.gatherv_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Gatherv<Backend>(input.data(), output.data(), send_counts, send_displs, root, comm, algo); },
      [&]() { Al::Gatherv<Backend>(output.data(), send_counts, send_displs, root, comm, algo); },
      [&]() { Al::NonblockingGatherv<Backend>(input.data(), output.data(), send_counts, send_displs, root, comm, req, algo); },
      [&]() { Al::NonblockingGatherv<Backend>(output.data(), send_counts, send_displs, root, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    std::vector<int> send_counts =
        Al::internal::mpi::intify_size_t_vector(this->get_options().send_counts);
    std::vector<int> send_displs =
        Al::internal::mpi::intify_size_t_vector(this->get_options().send_displs);
    // Account for in-place only being used at the root.
    void* sendbuf = (comm.rank() == root)
      ? this->buf_or_inplace(input.data())
      : (this->get_options().inplace ? output.data() : input.data());
    MPI_Gatherv(sendbuf,
                this->get_options().send_counts[comm.rank()],
                Al::internal::mpi::TypeMap<T>(),
                output.data(), send_counts.data(), send_displs.data(),
                Al::internal::mpi::TypeMap<T>(),
                root, comm.get_comm());
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return this->get_options().send_counts[comm.rank()];
    }
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& comm) {
    if (this->get_options().root == comm.rank()) {
      return std::accumulate(this->get_options().send_counts.begin(),
                             this->get_options().send_counts.end(), size_t{0});
    } else {
      if (this->get_options().inplace) {
        return this->get_options().send_counts[comm.rank()];
      } else {
        return 0;
      }
    }
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::reduce, Backend, T> :
  public OpRunnerShim<AlOperation::reduce, Backend, T,
                      OpRunner<AlOperation::reduce, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::reduce, Backend, T>;
  using OpRunnerShim<AlOperation::reduce, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::reduce,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    auto reduction_op = this->get_options().reduction_op;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.reduce_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Reduce<Backend>(input.data(), output.data(), input.size(), reduction_op, root, comm, algo); },
      [&]() { Al::Reduce<Backend>(output.data(), output.size(), reduction_op, root, comm, algo); },
      [&]() { Al::NonblockingReduce<Backend>(input.data(), output.data(), input.size(), reduction_op, root, comm, req, algo); },
      [&]() { Al::NonblockingReduce<Backend>(output.data(), output.size(), reduction_op, root, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op(
      this->get_options().reduction_op);
    // Account for in-place only being used at the root.
    void* sendbuf = (comm.rank() == root)
      ? this->buf_or_inplace(input.data())
      : (this->get_options().inplace ? output.data() : input.data());
    MPI_Reduce(sendbuf, output.data(),
               this->get_options().inplace ? output.size() : input.size(),
               Al::internal::mpi::TypeMap<T>(),
               reduction_op, root, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& /*comm*/) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return base_size;
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& comm) {
    if (this->get_options().root == comm.rank()) {
      return base_size;
    } else {
      if (this->get_options().inplace) {
        return base_size;
      } else {
        return 0;
      }
    }
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::reduce_scatter, Backend, T> :
  public OpRunnerShim<AlOperation::reduce_scatter, Backend, T,
                      OpRunner<AlOperation::reduce_scatter, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::reduce_scatter, Backend, T>;
  using OpRunnerShim<AlOperation::reduce_scatter, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::reduce_scatter,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    auto reduction_op = this->get_options().reduction_op;
    size_t size = this->get_options().inplace
      ? output.size() / comm.size()
      : input.size() / comm.size();
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.reduce_scatter_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Reduce_scatter<Backend>(input.data(), output.data(), size, reduction_op, comm, algo); },
      [&]() { Al::Reduce_scatter<Backend>(output.data(), size, reduction_op, comm, algo); },
      [&]() { Al::NonblockingReduce_scatter<Backend>(input.data(), output.data(), size, reduction_op, comm, req, algo); },
      [&]() { Al::NonblockingReduce_scatter<Backend>(output.data(), size, reduction_op, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op(
      this->get_options().reduction_op);
    MPI_Reduce_scatter_block(this->buf_or_inplace(input.data()), output.data(),
                             (this->get_options().inplace ? output.size() : input.size()) / comm.size(),
                             Al::internal::mpi::TypeMap<T>(),
                             reduction_op, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return base_size * comm.size();
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return base_size * comm.size();
    } else {
      return base_size;
    }
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::reduce_scatterv, Backend, T> :
  public OpRunnerShim<AlOperation::reduce_scatterv, Backend, T,
                      OpRunner<AlOperation::reduce_scatterv, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::reduce_scatterv, Backend, T>;
  using OpRunnerShim<AlOperation::reduce_scatterv, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::reduce_scatterv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    auto reduction_op = this->get_options().reduction_op;
    auto recv_counts = this->get_options().recv_counts;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.reduce_scatterv_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Reduce_scatterv<Backend>(input.data(), output.data(), recv_counts, reduction_op, comm, algo); },
      [&]() { Al::Reduce_scatterv<Backend>(output.data(), recv_counts, reduction_op, comm, algo); },
      [&]() { Al::NonblockingReduce_scatterv<Backend>(input.data(), output.data(), recv_counts, reduction_op, comm, req, algo); },
      [&]() { Al::NonblockingReduce_scatterv<Backend>(output.data(), recv_counts, reduction_op, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op(
      this->get_options().reduction_op);
    std::vector<int> counts = Al::internal::mpi::intify_size_t_vector(
      this->get_options().recv_counts);
    MPI_Reduce_scatter(this->buf_or_inplace(input.data()), output.data(),
                       counts.data(),
                       Al::internal::mpi::TypeMap<T>(),
                       reduction_op, comm.get_comm());
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& /*comm*/) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      return std::accumulate(this->get_options().recv_counts.begin(),
                             this->get_options().recv_counts.end(), size_t{0});
    }
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return std::accumulate(this->get_options().recv_counts.begin(),
                             this->get_options().recv_counts.end(), size_t{0});
    } else {
      return this->get_options().recv_counts[comm.rank()];
    }
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::scatter, Backend, T> :
  public OpRunnerShim<AlOperation::scatter, Backend, T,
                      OpRunner<AlOperation::scatter, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::scatter, Backend, T>;
  using OpRunnerShim<AlOperation::scatter, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::scatter,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    size_t size = (comm.rank() == root)
      ? (this->get_options().inplace ? output.size() : input.size()) / comm.size()
      : output.size();
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.scatter_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Scatter<Backend>(input.data(), output.data(), size, root, comm, algo); },
      [&]() { Al::Scatter<Backend>(output.data(), size, root, comm, algo); },
      [&]() { Al::NonblockingScatter<Backend>(input.data(), output.data(), size, root, comm, req, algo); },
      [&]() { Al::NonblockingScatter<Backend>(output.data(), size, root, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    size_t size = (comm.rank() == root)
      ? (this->get_options().inplace ? output.size() : input.size()) / comm.size()
      : output.size();
    // Account for in-place needing to be passed as the recvbuf on the root.
    void* sendbuf = (comm.rank() == root)
      ? (this->get_options().inplace ? output.data() : input.data())
      : input.data();
    void* recvbuf = (comm.rank() == root)
      ? this->buf_or_inplace(output.data())
      : output.data();
    MPI_Scatter(sendbuf, size,
                Al::internal::mpi::TypeMap<T>(),
                recvbuf, size,
                Al::internal::mpi::TypeMap<T>(),
                root, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      if (this->get_options().root == comm.rank()) {
        return base_size * comm.size();
      } else {
        return 0;
      }
    }
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      if (this->get_options().root == comm.rank()) {
        return base_size * comm.size();
      } else {
        return base_size;
      }
    } else {
      return base_size;
    }
  }
};

template <typename Backend, typename T>
class OpRunner<AlOperation::scatterv, Backend, T> :
  public OpRunnerShim<AlOperation::scatterv, Backend, T,
                      OpRunner<AlOperation::scatterv, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::scatterv, Backend, T>;
  using OpRunnerShim<AlOperation::scatterv, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::scatterv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    auto send_counts = this->get_options().send_counts;
    auto send_displs = this->get_options().send_displs;
    typename Backend::req_type& req = this->get_options().req;
    auto algo = this->get_options().algos.scatterv_algo;
    this->inplace_nb_dispatch(
      [&]() { Al::Scatterv<Backend>(input.data(), output.data(), send_counts, send_displs, root, comm, algo); },
      [&]() { Al::Scatterv<Backend>(output.data(), send_counts, send_displs, root, comm, algo); },
      [&]() { Al::NonblockingScatterv<Backend>(input.data(), output.data(), send_counts, send_displs, root, comm, req, algo); },
      [&]() { Al::NonblockingScatterv<Backend>(output.data(), send_counts, send_displs, root, comm, req, algo); });
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    std::vector<int> send_counts =
        Al::internal::mpi::intify_size_t_vector(this->get_options().send_counts);
    std::vector<int> send_displs =
        Al::internal::mpi::intify_size_t_vector(this->get_options().send_displs);
    // Account for in-place needing to be passed as the recvbuf on the root.
    void* sendbuf = (comm.rank() == root)
      ? (this->get_options().inplace ? output.data() : input.data())
      : input.data();
    void* recvbuf = (comm.rank() == root)
      ? this->buf_or_inplace(output.data())
      : output.data();
    MPI_Scatterv(sendbuf, send_counts.data(), send_displs.data(),
                 Al::internal::mpi::TypeMap<T>(),
                 recvbuf, send_counts[comm.rank()],
                 Al::internal::mpi::TypeMap<T>(),
                 root, comm.get_comm());
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      return 0;
    } else {
      if (this->get_options().root == comm.rank()) {
        return std::accumulate(this->get_options().send_counts.begin(),
                               this->get_options().send_counts.end(),
                               size_t{0});
      } else {
        return 0;
      }
    }
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      if (this->get_options().root == comm.rank()) {
        return std::accumulate(this->get_options().send_counts.begin(),
                               this->get_options().send_counts.end(),
                               size_t{0});
      } else {
        return this->get_options().send_counts[comm.rank()];
      }
    } else {
      return this->get_options().send_counts[comm.rank()];
    }
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::send, Backend, T> :
  public OpRunnerShim<AlOperation::send, Backend, T,
                      OpRunner<AlOperation::send, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::send, Backend, T>;
  using OpRunnerShim<AlOperation::send, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::send,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& /*output*/,
                typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      std::cerr << "No in-place for send" << std::endl;
      std::abort();
    }
    int dst = this->get_options().dst;
    typename Backend::req_type& req = this->get_options().req;
    this->inplace_nb_dispatch(
      [&]() { Al::Send<Backend>(input.data(), input.size(), dst, comm); },
      [&]() {},
      [&]() { Al::NonblockingSend<Backend>(input.data(), input.size(), dst, comm, req); },
      [&]() {});
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& /*output*/,
                    typename Backend::comm_type& comm) {
    MPI_Send(input.data(), input.size(),
             Al::internal::mpi::TypeMap<T>(),
             this->get_options().dst,
             0, comm.get_comm());
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& /*comm*/) {
    return base_size;
  }
  size_t get_output_size_impl(size_t /*base_size*/,
                              typename Backend::comm_type& /*comm*/) {
    return 0;
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::recv, Backend, T> :
  public OpRunnerShim<AlOperation::recv, Backend, T,
                      OpRunner<AlOperation::recv, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::recv, Backend, T>;
  using OpRunnerShim<AlOperation::recv, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::recv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& /*input*/,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      std::cerr << "No in-place for recv" << std::endl;
      std::abort();
    }
    int src = this->get_options().src;
    typename Backend::req_type& req = this->get_options().req;
    this->inplace_nb_dispatch(
      [&]() { Al::Recv<Backend>(output.data(), output.size(), src, comm); },
      [&]() {},
      [&]() { Al::NonblockingRecv<Backend>(output.data(), output.size(), src, comm, req); },
      [&]() {});
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& /*input*/,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Recv(output.data(), output.size(),
             Al::internal::mpi::TypeMap<T>(),
             this->get_options().src,
             0, comm.get_comm(), MPI_STATUS_IGNORE);
  }

  size_t get_input_size_impl(size_t /*base_size*/,
                             typename Backend::comm_type& /*comm*/) {
    return 0;
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& /*comm*/) {
    return base_size;
  }
};


template <typename Backend, typename T>
class OpRunner<AlOperation::sendrecv, Backend, T> :
  public OpRunnerShim<AlOperation::sendrecv, Backend, T,
                      OpRunner<AlOperation::sendrecv, Backend, T>> {
public:
  using ThisType = OpRunner<AlOperation::sendrecv, Backend, T>;
  using OpRunnerShim<AlOperation::sendrecv, Backend, T, ThisType>::OpRunnerShim;

  template <AlOperation Op2 = AlOperation::sendrecv,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      std::cerr << "No in-place for sendrecv" << std::endl;
      std::abort();
    }
    int src = this->get_options().src;
    int dst = this->get_options().dst;
    typename Backend::req_type& req = this->get_options().req;
    this->inplace_nb_dispatch(
      [&]() { Al::SendRecv<Backend>(input.data(), input.size(), dst, output.data(), output.size(), src, comm); },
      [&]() {},
      [&]() { Al::NonblockingSendRecv<Backend>(input.data(), input.size(), dst, output.data(), output.size(), src, comm, req); },
      [&]() {});
  }

  template <typename T2 = T,
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Sendrecv(input.data(), input.size(),
                 Al::internal::mpi::TypeMap<T>(),
                 this->get_options().dst, 0,
                 output.data(), output.size(),
                 Al::internal::mpi::TypeMap<T>(),
                 this->get_options().src, 0,
                 comm.get_comm(), MPI_STATUS_IGNORE);
  }

  size_t get_input_size_impl(size_t base_size,
                             typename Backend::comm_type& /*comm*/) {
    return base_size;
  }
  size_t get_output_size_impl(size_t base_size,
                              typename Backend::comm_type& /*comm*/) {
    return base_size;
  }
};
