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
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <vector>
#include <limits>
#include <type_traits>
#include <cstdlib>
#include "aluminum/traits/traits.hpp"
#include "test_utils.hpp"


/** Return true if str is a valid operator. */
bool is_operator_name(const std::string str) {
  static const std::unordered_set<std::string> ops = {
    "allgather",
    "allgatherv",
    "allreduce",
    "alltoall",
    "alltoallv",
    "barrier",
    "bcast",
    "gather",
    "gatherv",
    "reduce",
    "reduce_scatter",
    "reduce_scatterv",
    "scatter",
    "scatterv",
    "send",
    "recv",
    "sendrecv",
  };
  return ops.find(str) != ops.end();
}

/** Return the reduction operator corresponding to a string. */
Al::ReductionOperator get_reduction_op(const std::string redop_str) {
  static const std::unordered_map<std::string, Al::ReductionOperator> op_lookup = {
    {"sum", Al::ReductionOperator::sum},
    {"prod", Al::ReductionOperator::prod},
    {"min", Al::ReductionOperator::min},
    {"max", Al::ReductionOperator::max},
    {"lor", Al::ReductionOperator::lor},
    {"land", Al::ReductionOperator::land},
    {"lxor", Al::ReductionOperator::lxor},
    {"bor", Al::ReductionOperator::bor},
    {"band", Al::ReductionOperator::band},
    {"bxor", Al::ReductionOperator::bxor},
    {"avg", Al::ReductionOperator::avg},
  };
  auto i = op_lookup.find(redop_str);
  if (i == op_lookup.end()) {
    std::cerr << "Unsupported reduction operator" << redop_str << std::endl;
    std::abort();
  }
  return i->second;
}

// Traits for MPI type support.
template <typename T> struct IsTypeSupportedByMPI : std::false_type {};
template <> struct IsTypeSupportedByMPI<char> : std::true_type {};
template <> struct IsTypeSupportedByMPI<signed char> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned char> : std::true_type {};
template <> struct IsTypeSupportedByMPI<short> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned short> : std::true_type {};
template <> struct IsTypeSupportedByMPI<int> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned int> : std::true_type {};
template <> struct IsTypeSupportedByMPI<long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<long long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<unsigned long long> : std::true_type {};
template <> struct IsTypeSupportedByMPI<float> : std::true_type {};
template <> struct IsTypeSupportedByMPI<double> : std::true_type {};
template <> struct IsTypeSupportedByMPI<long double> : std::true_type {};
#ifdef AL_HAS_HALF
template <> struct IsTypeSupportedByMPI<__half> : std::true_type {};
#endif
#ifdef AL_HAS_BFLOAT
template <> struct IsTypeSupportedByMPI<al_bfloat16> : std::true_type {};
#endif

// Algorithms and string name supported by a particular backend/operator.
template <Al::AlOperation Op, typename Backend>
std::vector<std::pair<std::string, typename Al::OpAlgoType<Op, Backend>::type>> get_supported_algos() {
  return {{"automatic", Al::OpAlgoType<Op, Backend>::type::automatic}};
}

// Backend names.
template <typename Backend> constexpr char AlBackendName[] = "unknown";


/** Helper to call a functor with the right op as a template parameter. */
template <typename F>
auto call_op_functor(Al::AlOperation op, F functor) {
  switch (op) {
  case Al::AlOperation::allgather:
    return functor.template operator()<Al::AlOperation::allgather>();
  case Al::AlOperation::allgatherv:
    return functor.template operator()<Al::AlOperation::allgatherv>();
  case Al::AlOperation::allreduce:
    return functor.template operator()<Al::AlOperation::allreduce>();
  case Al::AlOperation::alltoall:
    return functor.template operator()<Al::AlOperation::alltoall>();
  case Al::AlOperation::alltoallv:
    return functor.template operator()<Al::AlOperation::alltoallv>();
    case Al::AlOperation::barrier:
    return functor.template operator()<Al::AlOperation::barrier>();
  case Al::AlOperation::bcast:
    return functor.template operator()<Al::AlOperation::bcast>();
  case Al::AlOperation::gather:
    return functor.template operator()<Al::AlOperation::gather>();
  case Al::AlOperation::gatherv:
    return functor.template operator()<Al::AlOperation::gatherv>();
  case Al::AlOperation::reduce:
    return functor.template operator()<Al::AlOperation::reduce>();
  case Al::AlOperation::reduce_scatter:
    return functor.template operator()<Al::AlOperation::reduce_scatter>();
  case Al::AlOperation::reduce_scatterv:
    return functor.template operator()<Al::AlOperation::reduce_scatterv>();
  case Al::AlOperation::scatter:
    return functor.template operator()<Al::AlOperation::scatter>();
  case Al::AlOperation::scatterv:
    return functor.template operator()<Al::AlOperation::scatterv>();
  case Al::AlOperation::send:
    return functor.template operator()<Al::AlOperation::send>();
  case Al::AlOperation::recv:
    return functor.template operator()<Al::AlOperation::recv>();
  case Al::AlOperation::sendrecv:
    return functor.template operator()<Al::AlOperation::sendrecv>();
  default:
    std::cerr << "Unknown AlOperation" << std::endl;
    std::abort();
  }
}

/** Return the AlOperation enum corresponding to op_str. */
Al::AlOperation op_str_to_op(const std::string op_str) {
  static const std::unordered_map<std::string, Al::AlOperation> lookup = {
    {"allgather", Al::AlOperation::allgather},
    {"allgatherv", Al::AlOperation::allgatherv},
    {"allreduce", Al::AlOperation::allreduce},
    {"alltoall", Al::AlOperation::alltoall},
    {"alltoallv", Al::AlOperation::alltoallv},
    {"barrier", Al::AlOperation::barrier},
    {"bcast", Al::AlOperation::bcast},
    {"gather", Al::AlOperation::gather},
    {"gatherv", Al::AlOperation::gatherv},
    {"reduce", Al::AlOperation::reduce},
    {"reduce_scatter", Al::AlOperation::reduce_scatter},
    {"reduce_scatterv", Al::AlOperation::reduce_scatterv},
    {"scatter", Al::AlOperation::scatter},
    {"scatterv", Al::AlOperation::scatterv},
    {"send", Al::AlOperation::send},
    {"recv", Al::AlOperation::recv},
    {"sendrecv", Al::AlOperation::sendrecv},
  };
  auto i = lookup.find(op_str);
  if (i == lookup.end()) {
    std::cerr << "Unknown operator " << op_str << std::endl;
    std::abort();
  }
  return i->second;
}

struct op_name_functor {
  template <Al::AlOperation Op>
  std::string operator()() {
    return Al::AlOperationName<Op>;
  }
};
std::string op_name(Al::AlOperation op) {
  return call_op_functor(op, op_name_functor());
}

template <typename Backend>
struct op_supported_functor {
  template <Al::AlOperation Op>
  bool operator()() {
    return Al::IsOpSupported<Op, Backend>::value;
  }
};
/** Return true if the operator is supported by the backend. */
template <typename Backend>
bool is_op_supported(Al::AlOperation op) {
  return call_op_functor(op, op_supported_functor<Backend>());
}

struct reduction_op_functor {
  template <Al::AlOperation Op>
  bool operator()() {
    return Al::IsReductionOp<Op>::value;
  }
};
/** Return true if the operator takes a reduction operator. */
bool requires_reduction_op(Al::AlOperation op) {
  return call_op_functor(op, reduction_op_functor());
}

struct vector_op_functor {
  template <Al::AlOperation Op>
  bool operator()() {
    return Al::IsVectorOp<Op>::value;
  }
};
/** Return true if the operator is a vector operator. */
bool is_vector_op(Al::AlOperation op) {
  return call_op_functor(op, vector_op_functor());
}

struct collective_op_functor {
  template <Al::AlOperation Op>
  bool operator()() {
    return Al::IsCollectiveOp<Op>::value;
  }
};
/** Return true if the operator is a collective operation. */
bool is_collective_op(Al::AlOperation op) {
  return call_op_functor(op, collective_op_functor());
}

struct pt2pt_op_functor {
  template <Al::AlOperation Op>
  bool operator()() {
    return Al::IsPt2PtOp<Op>::value;
  }
};
/** Return true if the operator is a point-to-point operation. */
bool is_pt2pt_op(Al::AlOperation op) {
  return call_op_functor(op, pt2pt_op_functor());
}

/** Return true if the reduction operator is supported by the backend. */
template <typename Backend>
bool is_reduction_operator_supported(Al::ReductionOperator op) {
  static const std::unordered_map<Al::ReductionOperator, bool> op_support = {
    {Al::ReductionOperator::sum,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::sum>::value},
    {Al::ReductionOperator::prod,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::prod>::value},
    {Al::ReductionOperator::min,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::min>::value},
    {Al::ReductionOperator::max,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::max>::value},
    {Al::ReductionOperator::lor,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::lor>::value},
    {Al::ReductionOperator::land,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::land>::value},
    {Al::ReductionOperator::lxor,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::lxor>::value},
    {Al::ReductionOperator::bor,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::bor>::value},
    {Al::ReductionOperator::band,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::band>::value},
    {Al::ReductionOperator::bxor,
     Al::IsReductionOpSupported<Backend, Al::ReductionOperator::bxor>::value},
  };
  auto i = op_support.find(op);
  if (i == op_support.end()) {
    std::cerr << "Unknown reduction operator" << std::endl;
    std::abort();
  }
  return i->second;
}

struct supports_algos_functor {
  template <Al::AlOperation Op>
  bool operator()() {
    return Al::OpSupportsAlgos<Op>::value;
  }
};
/** Return true if the operator supports different algorithms. */
bool op_supports_algos(Al::AlOperation op) {
  return call_op_functor(op, supports_algos_functor());
}

/** Contains algorithms for all supported operations. */
template <typename Backend> struct AlgorithmOptions {};

/** Helpers to set an algorithm for a particular op. */
template <Al::AlOperation Op, typename Backend> struct AlgoAccessor {};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::allgather, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::allgather, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.allgather_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::allgather, Backend>::type algo) {
    algo_opts.allgather_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::allgatherv, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::allgatherv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.allgatherv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::allgatherv, Backend>::type algo) {
    algo_opts.allgatherv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::allreduce, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::allreduce, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.allreduce_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::allreduce, Backend>::type algo) {
    algo_opts.allreduce_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::alltoall, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::alltoall, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.alltoall_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::alltoall, Backend>::type algo) {
    algo_opts.alltoall_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::alltoallv, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::alltoallv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.alltoallv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::alltoallv, Backend>::type algo) {
    algo_opts.alltoallv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::barrier, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::barrier, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.barrier_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::barrier, Backend>::type algo) {
    algo_opts.barrier_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::bcast, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::bcast, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.bcast_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::bcast, Backend>::type algo) {
    algo_opts.bcast_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::gather, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::gather, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.gather_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::gather, Backend>::type algo) {
    algo_opts.gather_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::gatherv, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::gatherv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.gatherv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::gatherv, Backend>::type algo) {
    algo_opts.gatherv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::reduce, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::reduce, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.reduce_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::reduce, Backend>::type algo) {
    algo_opts.reduce_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::reduce_scatter, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::reduce_scatter, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.reduce_scatter_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::reduce_scatter, Backend>::type algo) {
    algo_opts.reduce_scatter_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::reduce_scatterv, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::reduce_scatterv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.reduce_scatterv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::reduce_scatterv, Backend>::type algo) {
    algo_opts.reduce_scatterv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::scatter, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::scatter, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.scatter_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::scatter, Backend>::type algo) {
    algo_opts.scatter_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<Al::AlOperation::scatterv, Backend> {
  typename Al::OpAlgoType<Al::AlOperation::scatterv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.scatterv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename Al::OpAlgoType<Al::AlOperation::scatterv, Backend>::type algo) {
    algo_opts.scatterv_algo = algo;
  }
};

template <typename Backend>
struct get_algorithms_functor {
  std::string algo;
  get_algorithms_functor(std::string algo_) : algo(algo_) {}

  template <Al::AlOperation Op,
            std::enable_if_t<Al::IsOpSupported<Op, Backend>::value && Al::OpSupportsAlgos<Op>::value, bool> = true>
  std::vector<AlgorithmOptions<Backend>> operator()() {
    auto supported_algos = get_supported_algos<Op, Backend>();
    std::vector<AlgorithmOptions<Backend>> algos;
    AlgoAccessor<Op, Backend> setter;
    for (const auto& p : supported_algos) {
      if (algo == "all" || p.first == algo) {
        AlgorithmOptions<Backend> algo_opts;
        setter.set(algo_opts, p.second);
        algos.push_back(algo_opts);
      }
    }
    return algos;
  }
  template <Al::AlOperation Op,
            std::enable_if_t<!Al::IsOpSupported<Op, Backend>::value || !Al::OpSupportsAlgos<Op>::value, bool> = true>
  std::vector<AlgorithmOptions<Backend>> operator()() {
    return {};
  }
};

/**
 * Return a vector of AlgorithmOptions corresponding to what is requested
 * in algo.
 *
 * If algo is "all", all supported algorithms are returned.
 * If algo is an empty string, the automatic algorithm is returned.
 */
template <typename Backend>
std::vector<AlgorithmOptions<Backend>> get_algorithms(
  Al::AlOperation op, std::string algo) {
  if (algo == "") {
    algo = "automatic";
  }
  return call_op_functor(op, get_algorithms_functor<Backend>(algo));
}

/** Pass options that may be used by an operator. */
template <typename Backend>
struct OpOptions {
  bool inplace = false;
  bool nonblocking = false;
  int root = 0;
  int src = -1;
  int dst = -1;
  std::vector<size_t> send_counts = {};
  std::vector<size_t> send_displs = {};
  std::vector<size_t> recv_counts = {};
  std::vector<size_t> recv_displs = {};
  Al::ReductionOperator reduction_op = Al::ReductionOperator::sum;
  typename Backend::req_type req = Backend::null_req;
  AlgorithmOptions<Backend> algos;
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
            std::enable_if_t<IsTypeSupportedByMPI<T2>::value, bool> = true>
  void run_mpi_impl_int(std::vector<T>& input,
                        std::vector<T>& output,
                        typename Backend::comm_type& comm) {
    static_cast<Child*>(this)->run_mpi_impl(input, output, comm);
  }
  template <typename T2 = T,
            std::enable_if_t<!IsTypeSupportedByMPI<T2>::value, bool> = true>
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

#include "op_runner_impl.hpp"


/**
 * Run an operator for a backend.
 */
template <typename Backend, typename T>
class OpDispatcher {
public:
  OpDispatcher(Al::AlOperation op_, OpOptions<Backend>& options_) :
    op(op_), options(options_) {
    if (!is_op_supported<Backend>(op)) {
      std::cerr << "Backend does not support operator " << op_name(op) << std::endl;
      std::abort();
    }
    if (!Al::IsTypeSupported<Backend, T>::value) {
      std::cerr << "Backend does not support type" << std::endl;
      std::abort();
    }
    if (requires_reduction_op(op)
        && !is_reduction_operator_supported<Backend>(options.reduction_op)) {
      std::cerr << "Backend does not support reduction operator" << std::endl;
      std::abort();
    }
  }

  ~OpDispatcher() {}

  /**
   * Run the operator.
   *
   * The result will be placed in output, which may be the same as data.
   */
  void run(
    typename VectorType<T, Backend>::type& input,
    typename VectorType<T, Backend>::type& output,
    typename Backend::comm_type& comm) {
    call_op_functor(op, run_dispatch_functor(input, output, comm, options));
  }

  /** Run the raw MPI version of the operator. */
  void run_mpi(
    std::vector<T>& input,
    std::vector<T>& output,
    typename Backend::comm_type& comm) {
    call_op_functor(op, run_mpi_dispatch_functor(input, output, comm, options));
  }

  const OpOptions<Backend> &get_options() const { return options; }

  Al::AlOperation get_op() const { return op; }

  /**
   * Return the needed input buffer size for an input of the given size.
   *
   * For vector operations, size is ignored and the counts in the given
   * options are used. Displacements are not accounted for.
   */
  size_t get_input_size(
    size_t base_size,
    typename Backend::comm_type& comm) const {
    return call_op_functor(op, input_size_functor(comm, base_size, options));
  }

  /**
   * Return the needed output buffer size for an input of given size.
   *
   * For vector operations, size is ignored and the counts in the given
   * options are used. Displacements are not accounted for.
   */
  size_t get_output_size(
    size_t base_size,
    typename Backend::comm_type& comm) const {
    return call_op_functor(op, output_size_functor(comm, base_size, options));
  }

protected:

  struct run_dispatch_functor {
    typename VectorType<T, Backend>::type& input;
    typename VectorType<T, Backend>::type& output;
    typename Backend::comm_type& comm;
    OpOptions<Backend>& opts;
    run_dispatch_functor(typename VectorType<T, Backend>::type& input_,
                         typename VectorType<T, Backend>::type& output_,
                         typename Backend::comm_type& comm_,
                         OpOptions<Backend>& opts_) :
      input(input_), output(output_), comm(comm_), opts(opts_) {};

    template <Al::AlOperation Op>
    void operator()() {
      OpRunner<Op, Backend, T> runner(opts);
      runner.run(input, output, comm);
    }
  };

  struct run_mpi_dispatch_functor {
    std::vector<T>& input;
    std::vector<T>& output;
    typename Backend::comm_type& comm;
    OpOptions<Backend>& opts;
    run_mpi_dispatch_functor(std::vector<T>& input_,
                             std::vector<T>& output_,
                             typename Backend::comm_type& comm_,
                             OpOptions<Backend>& opts_) :
      input(input_), output(output_), comm(comm_), opts(opts_) {};

    template <Al::AlOperation Op>
    void operator()() {
      OpRunner<Op, Backend, T> runner(opts);
      runner.run_mpi(input, output, comm);
    }
  };

  struct input_size_functor {
    typename Backend::comm_type& comm;
    size_t base_size;
    OpOptions<Backend>& opts;
    input_size_functor(typename Backend::comm_type& comm_,
                       size_t base_size_,
                       OpOptions<Backend>& opts_) :
      comm(comm_), base_size(base_size_), opts(opts_) {};

    template <Al::AlOperation Op>
    size_t operator()() {
      OpRunner<Op, Backend, T> runner(opts);
      return runner.get_input_size(base_size, comm);
    }
  };

  struct output_size_functor {
    typename Backend::comm_type& comm;
    size_t base_size;
    OpOptions<Backend>& opts;
    output_size_functor(typename Backend::comm_type& comm_,
                       size_t base_size_,
                       OpOptions<Backend>& opts_) :
      comm(comm_), base_size(base_size_), opts(opts_) {};

    template <Al::AlOperation Op>
    size_t operator()() {
      OpRunner<Op, Backend, T> runner(opts);
      return runner.get_output_size(base_size, comm);
    }
  };

private:
  Al::AlOperation op;
  OpOptions<Backend>& options;
};
