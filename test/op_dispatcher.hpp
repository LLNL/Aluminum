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
    {"bxor", Al::ReductionOperator::bxor}
  };
  auto i = op_lookup.find(redop_str);
  if (i == op_lookup.end()) {
    std::cerr << "Unsupported reduction operator" << redop_str << std::endl;
    std::abort();
  }
  return i->second;
}

/* Defines supported operations. */
enum class AlOperation {
  allgather, allgatherv, allreduce, alltoall, alltoallv, barrier, bcast,
  gather, gatherv, reduce, reduce_scatter, reduce_scatterv,
  scatter, scatterv, send, recv, sendrecv
};
template <AlOperation Op> constexpr char AlOperationName[] = "unknown";
template <> constexpr char AlOperationName<AlOperation::allgather>[] = "allgather";
template <> constexpr char AlOperationName<AlOperation::allgatherv>[] = "allgatherv";
template <> constexpr char AlOperationName<AlOperation::allreduce>[] = "allreduce";
template <> constexpr char AlOperationName<AlOperation::alltoall>[] = "alltoall";
template <> constexpr char AlOperationName<AlOperation::alltoallv>[] = "alltoallv";
template <> constexpr char AlOperationName<AlOperation::barrier>[] = "barrier";
template <> constexpr char AlOperationName<AlOperation::bcast>[] = "bcast";
template <> constexpr char AlOperationName<AlOperation::gather>[] = "gather";
template <> constexpr char AlOperationName<AlOperation::gatherv>[] = "gatherv";
template <> constexpr char AlOperationName<AlOperation::reduce>[] = "reduce";
template <> constexpr char AlOperationName<AlOperation::reduce_scatter>[] = "reduce_scatter";
template <> constexpr char AlOperationName<AlOperation::reduce_scatterv>[] = "reduce_scatterv";
template <> constexpr char AlOperationName<AlOperation::scatter>[] = "scatter";
template <> constexpr char AlOperationName<AlOperation::scatterv>[] = "scatterv";
template <> constexpr char AlOperationName<AlOperation::send>[] = "send";
template <> constexpr char AlOperationName<AlOperation::recv>[] = "recv";
template <> constexpr char AlOperationName<AlOperation::sendrecv>[] = "sendrecv";

// Traits for operator support.
template <AlOperation Op, typename Backend> struct IsOpSupported : std::false_type {};

// Traits for type support.
template <typename Backend, typename T>
struct IsTypeSupported : std::false_type {};

// Traits for whether an operator requires a reduction operator argument.
template <AlOperation Op> struct IsReductionOp : std::false_type {};
template <> struct IsReductionOp<AlOperation::allreduce> : std::true_type {};
template <> struct IsReductionOp<AlOperation::reduce> : std::true_type {};
template <> struct IsReductionOp<AlOperation::reduce_scatter> : std::true_type {};
template <> struct IsReductionOp<AlOperation::reduce_scatterv> : std::true_type {};

// Traits for whether an operator is a vector operator.
template <AlOperation Op> struct IsVectorOp : std::false_type {};
template <> struct IsVectorOp<AlOperation::allgatherv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::alltoallv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::gatherv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::reduce_scatterv> : std::true_type {};
template <> struct IsVectorOp<AlOperation::scatterv> : std::true_type {};

// Traits for whether an operator is a collective or point-to-point.
template <AlOperation Op> struct IsCollectiveOp : std::false_type {};
template <> struct IsCollectiveOp<AlOperation::allgather> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::allgatherv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::allreduce> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::alltoall> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::alltoallv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::barrier> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::bcast> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::gather> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::gatherv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::reduce> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::reduce_scatter> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::reduce_scatterv> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::scatter> : std::true_type {};
template <> struct IsCollectiveOp<AlOperation::scatterv> : std::true_type {};

template <AlOperation Op> struct IsPt2PtOp : std::false_type {};
template <> struct IsPt2PtOp<AlOperation::send> : std::true_type {};
template <> struct IsPt2PtOp<AlOperation::recv> : std::true_type {};
template <> struct IsPt2PtOp<AlOperation::sendrecv> : std::true_type {};

// Traits for whether an operator has a root.
template <AlOperation Op> struct IsRootedOp : std::false_type {};
template <> struct IsRootedOp<AlOperation::bcast> : std::true_type {};
template <> struct IsRootedOp<AlOperation::gather> : std::true_type {};
template <> struct IsRootedOp<AlOperation::gatherv> : std::true_type {};
template <> struct IsRootedOp<AlOperation::reduce> : std::true_type {};
template <> struct IsRootedOp<AlOperation::scatter> : std::true_type {};
template <> struct IsRootedOp<AlOperation::scatterv> : std::true_type {};

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

// Traits for reduction operator support.
template <typename Backend, Al::ReductionOperator op>
struct IsReductionOpSupported : std::false_type {};

// Traits for whether an operator supports different algorithms.
template <AlOperation Op> struct OpSupportsAlgos : std::false_type {};
template <> struct OpSupportsAlgos<AlOperation::allgather> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::allgatherv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::allreduce> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::alltoall> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::alltoallv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::bcast> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::barrier> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::gather> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::gatherv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::reduce> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::reduce_scatter> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::reduce_scatterv> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::scatter> : std::true_type {};
template <> struct OpSupportsAlgos<AlOperation::scatterv> : std::true_type {};

// Identify algorithm types for operators for a backend.
// TODO: It seems like this should be able to be generic for any backend
// because they all use the same naming pattern, but I couldn't figure out
// the right enable_if incantations.
template <AlOperation Op, typename Backend> struct OpAlgoType {};

// Algorithms and string name supported by a particular backend/operator.
template <AlOperation Op, typename Backend>
std::vector<std::pair<std::string, typename OpAlgoType<Op, Backend>::type>> get_supported_algos() {
  return {{"automatic", OpAlgoType<Op, Backend>::type::automatic}};
}

// Backend names.
template <typename Backend> constexpr char AlBackendName[] = "unknown";


/** Helper to call a functor with the right op as a template parameter. */
template <typename F>
auto call_op_functor(AlOperation op, F functor) {
  switch (op) {
  case AlOperation::allgather:
    return functor.template operator()<AlOperation::allgather>();
  case AlOperation::allgatherv:
    return functor.template operator()<AlOperation::allgatherv>();
  case AlOperation::allreduce:
    return functor.template operator()<AlOperation::allreduce>();
  case AlOperation::alltoall:
    return functor.template operator()<AlOperation::alltoall>();
  case AlOperation::alltoallv:
    return functor.template operator()<AlOperation::alltoallv>();
    case AlOperation::barrier:
    return functor.template operator()<AlOperation::barrier>();
  case AlOperation::bcast:
    return functor.template operator()<AlOperation::bcast>();
  case AlOperation::gather:
    return functor.template operator()<AlOperation::gather>();
  case AlOperation::gatherv:
    return functor.template operator()<AlOperation::gatherv>();
  case AlOperation::reduce:
    return functor.template operator()<AlOperation::reduce>();
  case AlOperation::reduce_scatter:
    return functor.template operator()<AlOperation::reduce_scatter>();
  case AlOperation::reduce_scatterv:
    return functor.template operator()<AlOperation::reduce_scatterv>();
  case AlOperation::scatter:
    return functor.template operator()<AlOperation::scatter>();
  case AlOperation::scatterv:
    return functor.template operator()<AlOperation::scatterv>();
  case AlOperation::send:
    return functor.template operator()<AlOperation::send>();
  case AlOperation::recv:
    return functor.template operator()<AlOperation::recv>();
  case AlOperation::sendrecv:
    return functor.template operator()<AlOperation::sendrecv>();
  default:
    std::cerr << "Unknown AlOperation" << std::endl;
    std::abort();
  }
}

/** Return the AlOperation enum corresponding to op_str. */
AlOperation op_str_to_op(const std::string op_str) {
  static const std::unordered_map<std::string, AlOperation> lookup = {
    {"allgather", AlOperation::allgather},
    {"allgatherv", AlOperation::allgatherv},
    {"allreduce", AlOperation::allreduce},
    {"alltoall", AlOperation::alltoall},
    {"alltoallv", AlOperation::alltoallv},
    {"barrier", AlOperation::barrier},
    {"bcast", AlOperation::bcast},
    {"gather", AlOperation::gather},
    {"gatherv", AlOperation::gatherv},
    {"reduce", AlOperation::reduce},
    {"reduce_scatter", AlOperation::reduce_scatter},
    {"reduce_scatterv", AlOperation::reduce_scatterv},
    {"scatter", AlOperation::scatter},
    {"scatterv", AlOperation::scatterv},
    {"send", AlOperation::send},
    {"recv", AlOperation::recv},
    {"sendrecv", AlOperation::sendrecv},
  };
  auto i = lookup.find(op_str);
  if (i == lookup.end()) {
    std::cerr << "Unknown operator " << op_str << std::endl;
    std::abort();
  }
  return i->second;
}

struct op_name_functor {
  template <AlOperation Op>
  std::string operator()() {
    return AlOperationName<Op>;
  }
};
std::string op_name(AlOperation op) {
  return call_op_functor(op, op_name_functor());
}

template <typename Backend>
struct op_supported_functor {
  template <AlOperation Op>
  bool operator()() {
    return IsOpSupported<Op, Backend>::value;
  }
};
/** Return true if the operator is supported by the backend. */
template <typename Backend>
bool is_op_supported(AlOperation op) {
  return call_op_functor(op, op_supported_functor<Backend>());
}

struct reduction_op_functor {
  template <AlOperation Op>
  bool operator()() {
    return IsReductionOp<Op>::value;
  }
};
/** Return true if the operator takes a reduction operator. */
bool requires_reduction_op(AlOperation op) {
  return call_op_functor(op, reduction_op_functor());
}

struct vector_op_functor {
  template <AlOperation Op>
  bool operator()() {
    return IsVectorOp<Op>::value;
  }
};
/** Return true if the operator is a vector operator. */
bool is_vector_op(AlOperation op) {
  return call_op_functor(op, vector_op_functor());
}

struct collective_op_functor {
  template <AlOperation Op>
  bool operator()() {
    return IsCollectiveOp<Op>::value;
  }
};
/** Return true if the operator is a collective operation. */
bool is_collective_op(AlOperation op) {
  return call_op_functor(op, collective_op_functor());
}

struct pt2pt_op_functor {
  template <AlOperation Op>
  bool operator()() {
    return IsPt2PtOp<Op>::value;
  }
};
/** Return true if the operator is a point-to-point operation. */
bool is_pt2pt_op(AlOperation op) {
  return call_op_functor(op, pt2pt_op_functor());
}

/** Return true if the reduction operator is supported by the backend. */
template <typename Backend>
bool is_reduction_operator_supported(Al::ReductionOperator op) {
  static const std::unordered_map<Al::ReductionOperator, bool> op_support = {
    {Al::ReductionOperator::sum,
     IsReductionOpSupported<Backend, Al::ReductionOperator::sum>::value},
    {Al::ReductionOperator::prod,
     IsReductionOpSupported<Backend, Al::ReductionOperator::prod>::value},
    {Al::ReductionOperator::min,
     IsReductionOpSupported<Backend, Al::ReductionOperator::min>::value},
    {Al::ReductionOperator::max,
     IsReductionOpSupported<Backend, Al::ReductionOperator::max>::value},
    {Al::ReductionOperator::lor,
     IsReductionOpSupported<Backend, Al::ReductionOperator::lor>::value},
    {Al::ReductionOperator::land,
     IsReductionOpSupported<Backend, Al::ReductionOperator::land>::value},
    {Al::ReductionOperator::lxor,
     IsReductionOpSupported<Backend, Al::ReductionOperator::lxor>::value},
    {Al::ReductionOperator::bor,
     IsReductionOpSupported<Backend, Al::ReductionOperator::bor>::value},
    {Al::ReductionOperator::band,
     IsReductionOpSupported<Backend, Al::ReductionOperator::band>::value},
    {Al::ReductionOperator::bxor,
     IsReductionOpSupported<Backend, Al::ReductionOperator::bxor>::value},
  };
  auto i = op_support.find(op);
  if (i == op_support.end()) {
    std::cerr << "Unknown reduction operator" << std::endl;
    std::abort();
  }
  return i->second;
}

struct supports_algos_functor {
  template <AlOperation Op>
  bool operator()() {
    return OpSupportsAlgos<Op>::value;
  }
};
/** Return true if the operator supports different algorithms. */
bool op_supports_algos(AlOperation op) {
  return call_op_functor(op, supports_algos_functor());
}

/** Contains algorithms for all supported operations. */
template <typename Backend> struct AlgorithmOptions {};

/** Helpers to set an algorithm for a particular op. */
template <AlOperation Op, typename Backend> struct AlgoAccessor {};
template <typename Backend> struct AlgoAccessor<AlOperation::allgather, Backend> {
  typename OpAlgoType<AlOperation::allgather, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.allgather_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::allgather, Backend>::type algo) {
    algo_opts.allgather_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::allgatherv, Backend> {
  typename OpAlgoType<AlOperation::allgatherv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.allgatherv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::allgatherv, Backend>::type algo) {
    algo_opts.allgatherv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::allreduce, Backend> {
  typename OpAlgoType<AlOperation::allreduce, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.allreduce_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::allreduce, Backend>::type algo) {
    algo_opts.allreduce_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::alltoall, Backend> {
  typename OpAlgoType<AlOperation::alltoall, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.alltoall_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::alltoall, Backend>::type algo) {
    algo_opts.alltoall_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::alltoallv, Backend> {
  typename OpAlgoType<AlOperation::alltoallv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.alltoallv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::alltoallv, Backend>::type algo) {
    algo_opts.alltoallv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::barrier, Backend> {
  typename OpAlgoType<AlOperation::barrier, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.barrier_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::barrier, Backend>::type algo) {
    algo_opts.barrier_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::bcast, Backend> {
  typename OpAlgoType<AlOperation::bcast, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.bcast_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::bcast, Backend>::type algo) {
    algo_opts.bcast_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::gather, Backend> {
  typename OpAlgoType<AlOperation::gather, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.gather_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::gather, Backend>::type algo) {
    algo_opts.gather_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::gatherv, Backend> {
  typename OpAlgoType<AlOperation::gatherv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.gatherv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::gatherv, Backend>::type algo) {
    algo_opts.gatherv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::reduce, Backend> {
  typename OpAlgoType<AlOperation::reduce, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.reduce_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::reduce, Backend>::type algo) {
    algo_opts.reduce_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::reduce_scatter, Backend> {
  typename OpAlgoType<AlOperation::reduce_scatter, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.reduce_scatter_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::reduce_scatter, Backend>::type algo) {
    algo_opts.reduce_scatter_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::reduce_scatterv, Backend> {
  typename OpAlgoType<AlOperation::reduce_scatterv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.reduce_scatterv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::reduce_scatterv, Backend>::type algo) {
    algo_opts.reduce_scatterv_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::scatter, Backend> {
  typename OpAlgoType<AlOperation::scatter, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.scatter_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::scatter, Backend>::type algo) {
    algo_opts.scatter_algo = algo;
  }
};
template <typename Backend> struct AlgoAccessor<AlOperation::scatterv, Backend> {
  typename OpAlgoType<AlOperation::scatterv, Backend>::type get(
    AlgorithmOptions<Backend>& algo_opts) {
    return algo_opts.scatterv_algo;
  }
  void set(AlgorithmOptions<Backend>& algo_opts,
           typename OpAlgoType<AlOperation::scatterv, Backend>::type algo) {
    algo_opts.scatterv_algo = algo;
  }
};

template <typename Backend>
struct get_algorithms_functor {
  std::string algo;
  get_algorithms_functor(std::string algo_) : algo(algo_) {}

  template <AlOperation Op,
            std::enable_if_t<IsOpSupported<Op, Backend>::value && OpSupportsAlgos<Op>::value, bool> = true>
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
  template <AlOperation Op,
            std::enable_if_t<!IsOpSupported<Op, Backend>::value || !OpSupportsAlgos<Op>::value, bool> = true>
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
  AlOperation op, std::string algo) {
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
template <AlOperation Op, typename Backend, typename T, typename Child>
class OpRunnerShim : public OpRunnerBase<Backend, T, OpRunnerShim<Op, Backend, T, Child>> {
public:
  using OpRunnerBase<Backend, T, OpRunnerShim<Op, Backend, T, Child>>::OpRunnerBase;

  std::string get_name_impl() const { return AlOperationName<Op>; }

  template <AlOperation Op2 = Op,
            std::enable_if_t<IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl_int(typename VectorType<T, Backend>::type& input,
                    typename VectorType<T, Backend>::type& output,
                    typename Backend::comm_type& comm) {
    static_cast<Child*>(this)->run_impl(input, output, comm);
  }
  template <AlOperation Op2 = Op,
            std::enable_if_t<!IsOpSupported<Op2, Backend>::value, bool> = true>
  void run_impl_int(typename VectorType<T, Backend>::type&,
                    typename VectorType<T, Backend>::type&,
                    typename Backend::comm_type&) {
    std::cerr << AlOperationName<Op> << " not supported by backend" << std::endl;
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
template <AlOperation Op, typename Backend, typename T>
class OpRunner : public OpRunnerShim<Op, Backend, T, OpRunner<Op, Backend, T>> {};

#include "op_runner_impl.hpp"


/**
 * Run an operator for a backend.
 */
template <typename Backend, typename T>
class OpDispatcher {
public:
  OpDispatcher(AlOperation op_, OpOptions<Backend>& options_) :
    op(op_), options(options_) {
    if (!is_op_supported<Backend>(op)) {
      std::cerr << "Backend does not support operator " << op_name(op) << std::endl;
      std::abort();
    }
    if (!IsTypeSupported<Backend, T>::value) {
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

    template <AlOperation Op>
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

    template <AlOperation Op>
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

    template <AlOperation Op>
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

    template <AlOperation Op>
    size_t operator()() {
      OpRunner<Op, Backend, T> runner(opts);
      return runner.get_output_size(base_size, comm);
    }
  };

private:
  AlOperation op;
  OpOptions<Backend>& options;
};
