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
#include "aluminum/traits/traits.hpp"
#include "algo_support.hpp"
#include "test_utils.hpp"

/** Pass options that may be used by an operator. */
template <typename Backend>
struct OpOptions {
  bool inplace = false;
  bool nonblocking = false;
  int root = 0;
  int src = -1;
  std::vector<int> srcs = {};
  int dst = -1;
  std::vector<int> dests = {};
  std::vector<size_t> send_counts = {};
  std::vector<size_t> send_displs = {};
  std::vector<size_t> recv_counts = {};
  std::vector<size_t> recv_displs = {};
  Al::ReductionOperator reduction_op = Al::ReductionOperator::sum;
  typename Backend::req_type req = Backend::null_req;
  AlgorithmOptions<Backend> algos;
  bool register_memory = false;
};

/** Abstract base class for running an operator. */
template <typename Backend, typename T, typename Child>
class OpRunnerBase {
public:
  OpRunnerBase(OpOptions<Backend> &options_) :
    options(options_) {}
  ~OpRunnerBase() {}

  const OpOptions<Backend>& get_options() const { return options; }
  OpOptions<Backend>& get_options() { return options; }
  std::string get_name() const {
    return static_cast<const Child*>(this)->get_name_impl();
  };

  void run(typename VectorType<T, Backend>::type& input,
           typename VectorType<T, Backend>::type& output,
           typename Backend::comm_type& comm) {
    static_cast<Child*>(this)->run_impl_int(input, output, comm);
  }
  void run_mpi(std::vector<T>& input,
               std::vector<T>& output,
               typename Backend::comm_type& comm) {
    static_cast<Child*>(this)->run_mpi_impl_int(input, output, comm);
  }

  size_t get_input_size(size_t base_size,
                        typename Backend::comm_type& comm) {
    return static_cast<Child*>(this)->get_input_size_int_impl(base_size, comm);
  }
  size_t get_output_size(size_t base_size,
                         typename Backend::comm_type& comm) {
    return static_cast<Child*>(this)->get_output_size_int_impl(base_size, comm);
  }

protected:
  void inplace_nb_dispatch(std::function<void()> nip_b,
                           std::function<void()> ip_b,
                           std::function<void()> nip_nb,
                           std::function<void()> ip_nb) {
    if (get_options().inplace) {
      if (get_options().nonblocking) {
        ip_nb();
      } else {
        ip_b();
      }
    } else {
      if (get_options().nonblocking) {
        nip_nb();
      } else {
        nip_b();
      }
    }
  }

  void* buf_or_inplace(T* buf) {
    return get_options().inplace ? MPI_IN_PLACE : buf;
  }

private:
  OpOptions<Backend>& options;
};

/**
 * Intermediate ABC providing some common functions.
 */
template <Al::AlOperation Op, typename Backend, typename T, typename Child>
class OpRunnerShim : public OpRunnerBase<Backend, T, OpRunnerShim<Op, Backend, T, Child>> {
public:
  using OpRunnerBase<Backend, T, OpRunnerShim<Op, Backend, T, Child>>::OpRunnerBase;

  std::string get_name_impl() const { return Al::AlOperationName<Op>; }

  template <Al::AlOperation Op2 = Op,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl_int(typename VectorType<T, Backend>::type& input,
                    typename VectorType<T, Backend>::type& output,
                    typename Backend::comm_type& comm) {
    static_cast<Child*>(this)->run_impl(input, output, comm);
  }
  template <Al::AlOperation Op2 = Op,
            std::enable_if_t<!Al::IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl_int(typename VectorType<T, Backend>::type&,
                    typename VectorType<T, Backend>::type&,
                    typename Backend::comm_type&) {
    std::cerr << Al::AlOperationName<Op> << " not supported by backend" << std::endl;
    std::abort();
  }

  template <typename T2 = T,
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl_int(std::vector<T>& input,
                        std::vector<T>& output,
                        typename Backend::comm_type& comm) {
    static_cast<Child*>(this)->run_mpi_impl(input, output, comm);
  }
  template <typename T2 = T,
            std::enable_if_t<!Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl_int(std::vector<T>&, std::vector<T>&,
                        typename Backend::comm_type&) {
    std::cerr << "Type not supported by MPI" << std::endl;
    std::abort();
  }

  size_t get_input_size_int_impl(size_t base_size,
                                 typename Backend::comm_type& comm) {
    return static_cast<Child*>(this)->get_input_size_impl(base_size, comm);
  }
  size_t get_output_size_int_impl(size_t base_size,
                                  typename Backend::comm_type& comm) {
    return static_cast<Child*>(this)->get_output_size_impl(base_size, comm);
  }

};

/**
 * Main OpRunner class, handles calls to Aluminum/MPI for an operator.
 *
 * Should be specialized for specific operators.
 */
template <Al::AlOperation Op, typename Backend, typename T>
class OpRunner : public OpRunnerShim<Op, Backend, T, OpRunner<Op, Backend, T>> {};

// Specific implementations are below:

template <typename Backend, typename T>
class OpRunner<Al::AlOperation::allgather, Backend, T> :
  public OpRunnerShim<Al::AlOperation::allgather, Backend, T,
                      OpRunner<Al::AlOperation::allgather, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::allgather, Backend, T>;
  using OpRunnerShim<Al::AlOperation::allgather, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::allgather,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Allgather)(
      this->buf_or_inplace(input.data()), input.size(),
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
class OpRunner<Al::AlOperation::allgatherv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::allgatherv, Backend, T,
                      OpRunner<Al::AlOperation::allgatherv, Backend, T>> {
public:
  using OpRunnerShim<Al::AlOperation::allgatherv, Backend, T,
                     OpRunner<Al::AlOperation::allgatherv, Backend, T>>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::allgatherv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    auto counts = Al::internal::mpi::countify_size_t_vector(
      this->get_options().send_counts);
    auto displs = Al::internal::mpi::displify_size_t_vector(
      this->get_options().send_displs);
    AL_MPI_LARGE_COUNT_CALL(MPI_Allgatherv)(
      this->buf_or_inplace(input.data()), counts[comm.rank()],
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
class OpRunner<Al::AlOperation::allreduce, Backend, T> :
  public OpRunnerShim<Al::AlOperation::allreduce, Backend, T,
                      OpRunner<Al::AlOperation::allreduce, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::allreduce, Backend, T>;
  using OpRunnerShim<Al::AlOperation::allreduce, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::allreduce,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op<T>(
      this->get_options().reduction_op);
    AL_MPI_LARGE_COUNT_CALL(MPI_Allreduce)(
      this->buf_or_inplace(input.data()), output.data(),
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
class OpRunner<Al::AlOperation::alltoall, Backend, T> :
  public OpRunnerShim<Al::AlOperation::alltoall, Backend, T,
                      OpRunner<Al::AlOperation::alltoall, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::alltoall, Backend, T>;
  using OpRunnerShim<Al::AlOperation::alltoall, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::alltoall,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    size_t size = this->get_options().inplace
      ? output.size() / comm.size()
      : input.size() / comm.size();
    AL_MPI_LARGE_COUNT_CALL(MPI_Alltoall)(
      this->buf_or_inplace(input.data()), size,
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
class OpRunner<Al::AlOperation::alltoallv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::alltoallv, Backend, T,
                      OpRunner<Al::AlOperation::alltoallv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::alltoallv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::alltoallv, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::alltoallv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    auto send_counts =
        Al::internal::mpi::countify_size_t_vector(this->get_options().send_counts);
    auto send_displs =
        Al::internal::mpi::displify_size_t_vector(this->get_options().send_displs);
    auto recv_counts =
        Al::internal::mpi::countify_size_t_vector(this->get_options().recv_counts);
    auto recv_displs =
        Al::internal::mpi::displify_size_t_vector(this->get_options().recv_displs);
    AL_MPI_LARGE_COUNT_CALL(MPI_Alltoallv)(
      this->buf_or_inplace(input.data()),
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
class OpRunner<Al::AlOperation::barrier, Backend, T> :
  public OpRunnerShim<Al::AlOperation::barrier, Backend, T,
                      OpRunner<Al::AlOperation::barrier, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::barrier, Backend, T>;
  using OpRunnerShim<Al::AlOperation::barrier, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::barrier,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
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
class OpRunner<Al::AlOperation::bcast, Backend, T> :
  public OpRunnerShim<Al::AlOperation::bcast, Backend, T,
                      OpRunner<Al::AlOperation::bcast, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::bcast, Backend, T>;
  using OpRunnerShim<Al::AlOperation::bcast, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::bcast,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& /*input*/,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Bcast)(
      output.data(), output.size(),
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
class OpRunner<Al::AlOperation::gather, Backend, T> :
  public OpRunnerShim<Al::AlOperation::gather, Backend, T,
                      OpRunner<Al::AlOperation::gather, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::gather, Backend, T>;
  using OpRunnerShim<Al::AlOperation::gather, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::gather,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
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
    AL_MPI_LARGE_COUNT_CALL(MPI_Gather)(
      sendbuf, size,
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
class OpRunner<Al::AlOperation::gatherv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::gatherv, Backend, T,
                      OpRunner<Al::AlOperation::gatherv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::gatherv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::gatherv, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::gatherv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    auto send_counts =
        Al::internal::mpi::countify_size_t_vector(this->get_options().send_counts);
    auto send_displs =
        Al::internal::mpi::displify_size_t_vector(this->get_options().send_displs);
    // Account for in-place only being used at the root.
    void* sendbuf = (comm.rank() == root)
      ? this->buf_or_inplace(input.data())
      : (this->get_options().inplace ? output.data() : input.data());
    AL_MPI_LARGE_COUNT_CALL(MPI_Gatherv)(
      sendbuf,
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
class OpRunner<Al::AlOperation::reduce, Backend, T> :
  public OpRunnerShim<Al::AlOperation::reduce, Backend, T,
                      OpRunner<Al::AlOperation::reduce, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::reduce, Backend, T>;
  using OpRunnerShim<Al::AlOperation::reduce, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::reduce,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op<T>(
      this->get_options().reduction_op);
    // Account for in-place only being used at the root.
    void* sendbuf = (comm.rank() == root)
      ? this->buf_or_inplace(input.data())
      : (this->get_options().inplace ? output.data() : input.data());
    AL_MPI_LARGE_COUNT_CALL(MPI_Reduce)(
      sendbuf, output.data(),
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
class OpRunner<Al::AlOperation::reduce_scatter, Backend, T> :
  public OpRunnerShim<Al::AlOperation::reduce_scatter, Backend, T,
                      OpRunner<Al::AlOperation::reduce_scatter, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::reduce_scatter, Backend, T>;
  using OpRunnerShim<Al::AlOperation::reduce_scatter, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::reduce_scatter,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op<T>(
      this->get_options().reduction_op);
    AL_MPI_LARGE_COUNT_CALL(MPI_Reduce_scatter_block)(
      this->buf_or_inplace(input.data()), output.data(),
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
class OpRunner<Al::AlOperation::reduce_scatterv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::reduce_scatterv, Backend, T,
                      OpRunner<Al::AlOperation::reduce_scatterv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::reduce_scatterv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::reduce_scatterv, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::reduce_scatterv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    MPI_Op reduction_op = Al::internal::mpi::ReductionOperator2MPI_Op<T>(
      this->get_options().reduction_op);
    auto counts = Al::internal::mpi::countify_size_t_vector(
      this->get_options().recv_counts);
    AL_MPI_LARGE_COUNT_CALL(MPI_Reduce_scatter)(
      this->buf_or_inplace(input.data()), output.data(),
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
class OpRunner<Al::AlOperation::scatter, Backend, T> :
  public OpRunnerShim<Al::AlOperation::scatter, Backend, T,
                      OpRunner<Al::AlOperation::scatter, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::scatter, Backend, T>;
  using OpRunnerShim<Al::AlOperation::scatter, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::scatter,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
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
    AL_MPI_LARGE_COUNT_CALL(MPI_Scatter)(
      sendbuf, size,
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
class OpRunner<Al::AlOperation::scatterv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::scatterv, Backend, T,
                      OpRunner<Al::AlOperation::scatterv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::scatterv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::scatterv, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::scatterv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    int root = this->get_options().root;
    auto send_counts =
        Al::internal::mpi::countify_size_t_vector(this->get_options().send_counts);
    auto send_displs =
        Al::internal::mpi::displify_size_t_vector(this->get_options().send_displs);
    // Account for in-place needing to be passed as the recvbuf on the root.
    void* sendbuf = (comm.rank() == root)
      ? (this->get_options().inplace ? output.data() : input.data())
      : input.data();
    void* recvbuf = (comm.rank() == root)
      ? this->buf_or_inplace(output.data())
      : output.data();
    AL_MPI_LARGE_COUNT_CALL(MPI_Scatterv)(
      sendbuf, send_counts.data(), send_displs.data(),
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
class OpRunner<Al::AlOperation::send, Backend, T> :
  public OpRunnerShim<Al::AlOperation::send, Backend, T,
                      OpRunner<Al::AlOperation::send, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::send, Backend, T>;
  using OpRunnerShim<Al::AlOperation::send, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::send,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& /*output*/,
                    typename Backend::comm_type& comm) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Send)(
      input.data(), input.size(),
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
class OpRunner<Al::AlOperation::recv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::recv, Backend, T,
                      OpRunner<Al::AlOperation::recv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::recv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::recv, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::recv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
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
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& /*input*/,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    AL_MPI_LARGE_COUNT_CALL(MPI_Recv)(
      output.data(), output.size(),
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
class OpRunner<Al::AlOperation::sendrecv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::sendrecv, Backend, T,
                      OpRunner<Al::AlOperation::sendrecv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::sendrecv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::sendrecv, Backend, T, ThisType>::OpRunnerShim;

  template <Al::AlOperation Op2 = Al::AlOperation::sendrecv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    int src = this->get_options().src;
    int dst = this->get_options().dst;
    typename Backend::req_type& req = this->get_options().req;
    this->inplace_nb_dispatch(
      [&]() { Al::SendRecv<Backend>(input.data(), input.size(), dst, output.data(), output.size(), src, comm); },
      [&]() { Al::SendRecv<Backend>(output.data(), output.size(), dst, src, comm); },
      [&]() { Al::NonblockingSendRecv<Backend>(input.data(), input.size(), dst, output.data(), output.size(), src, comm, req); },
      [&]() { Al::NonblockingSendRecv<Backend>(output.data(), output.size(), dst, src, comm, req); });
  }

  template <typename T2 = T,
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input,
                    std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    if (this->get_options().inplace) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Sendrecv_replace)(
        output.data(), output.size(),
        Al::internal::mpi::TypeMap<T>(),
        this->get_options().dst, 0,
        this->get_options().src, 0,
        comm.get_comm(), MPI_STATUS_IGNORE);
    } else {
      AL_MPI_LARGE_COUNT_CALL(MPI_Sendrecv)(
        input.data(), input.size(),
        Al::internal::mpi::TypeMap<T>(),
        this->get_options().dst, 0,
        output.data(), output.size(),
        Al::internal::mpi::TypeMap<T>(),
        this->get_options().src, 0,
        comm.get_comm(), MPI_STATUS_IGNORE);
    }
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

template <typename Backend, typename T>
class OpRunner<Al::AlOperation::multisendrecv, Backend, T> :
  public OpRunnerShim<Al::AlOperation::multisendrecv, Backend, T,
                      OpRunner<Al::AlOperation::multisendrecv, Backend, T>> {
public:
  using ThisType = OpRunner<Al::AlOperation::multisendrecv, Backend, T>;
  using OpRunnerShim<Al::AlOperation::multisendrecv, Backend, T,
                     ThisType>::OpRunnerShim;

  // Note: For simplicity, this uses the send counts to compute the
  // offsets in the input/output buffers.

  template <Al::AlOperation Op2 = Al::AlOperation::multisendrecv,
            std::enable_if_t<Al::IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl(typename VectorType<T, Backend>::type& input,
                typename VectorType<T, Backend>::type& output,
                typename Backend::comm_type& comm) {
    auto srcs = this->get_options().srcs;
    auto dests = this->get_options().dests;
    auto send_counts = this->get_options().send_counts;
    auto recv_counts = this->get_options().recv_counts;
    std::vector<const T*> send_buffers(this->get_options().inplace ? 0 : dests.size());
    std::vector<T*> recv_buffers(srcs.size());
    T* recv_buf = output.data();
    for (size_t i = 0; i < recv_buffers.size(); ++i) {
      recv_buffers[i] = recv_buf;
      recv_buf += recv_counts[i];
    }
    if (!this->get_options().inplace) {
      T* send_buf = input.data();
      for (size_t i = 0; i < send_buffers.size(); ++i) {
        send_buffers[i] = send_buf;
        send_buf += send_counts[i];
      }
    }
    typename Backend::req_type& req = this->get_options().req;
    this->inplace_nb_dispatch(
      [&]() { Al::MultiSendRecv<Backend>(send_buffers, send_counts, dests, recv_buffers, recv_counts, srcs, comm); },
      [&]() { Al::MultiSendRecv<Backend>(recv_buffers, recv_counts, dests, srcs, comm); },
      [&]() { Al::NonblockingMultiSendRecv<Backend>(send_buffers, send_counts, dests, recv_buffers, recv_counts, srcs, comm, req); },
      [&]() { Al::NonblockingMultiSendRecv<Backend>(recv_buffers, recv_counts, dests, srcs, comm, req); });
  }

  template <typename T2 = T,
            std::enable_if_t<Al::IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl(std::vector<T>& input, std::vector<T>& output,
                    typename Backend::comm_type& comm) {
    auto srcs = this->get_options().srcs;
    auto dests = this->get_options().dests;
    if (srcs.empty() && dests.empty()) {
      return;
    }
    auto send_counts =
      Al::internal::mpi::countify_size_t_vector(this->get_options().send_counts);
    auto recv_counts =
      Al::internal::mpi::displify_size_t_vector(this->get_options().recv_counts);
    std::vector<MPI_Request> reqs(srcs.size() + dests.size());
    T* recv_buf = output.data();
    T* send_buf = nullptr;
    std::vector<T> tmp_buf;
    if (this->get_options().inplace) {
      tmp_buf = output;
      send_buf = tmp_buf.data();
    } else {
      send_buf = input.data();
    }
    for (size_t i = 0; i < srcs.size(); ++i) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Irecv)(
        recv_buf, recv_counts[i], Al::internal::mpi::TypeMap<T>(),
        srcs[i], 0, comm.get_comm(), &reqs[i]);
      recv_buf += recv_counts[i];
    }
    for (size_t i = 0; i < dests.size(); ++i) {
      AL_MPI_LARGE_COUNT_CALL(MPI_Isend)(
        send_buf, send_counts[i], Al::internal::mpi::TypeMap<T>(),
        dests[i], 0, comm.get_comm(), &reqs[i + srcs.size()]);
      send_buf += send_counts[i];
    }
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
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
