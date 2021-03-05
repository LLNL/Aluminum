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

#include "test_utils.hpp"
#include "test_utils_mpi.hpp"
#include "cuda_vector.hpp"


template <typename T>
struct VectorType<T, Al::HostTransferBackend> {
  using type = CUDAVector<T>;

  static type gen_data(size_t count) {
    auto&& host_data = VectorType<T, Al::MPIBackend>::gen_data(count);
    CUDAVector<T> data(host_data);
    return data;
  }

  static std::vector<T> copy_to_host(const type& v) {
    return v.copyout();
  }
};

// Specialize to create a CUDA stream with the communicator.
template <>
CommWrapper<Al::HostTransferBackend>::CommWrapper(MPI_Comm mpi_comm) {
  cudaStream_t stream;
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamCreate(&stream));
  comm_ = std::make_unique<typename Al::HostTransferBackend::comm_type>(
    mpi_comm, stream);
}
template <>
CommWrapper<Al::HostTransferBackend>::~CommWrapper() noexcept(false) {
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamDestroy(comm_->get_stream()));
}

template <>
void complete_operations<Al::HostTransferBackend>(
  typename Al::HostTransferBackend::comm_type& comm) {
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamSynchronize(comm.get_stream()));
}

// Operator support.
template <> struct IsOpSupported<AlOperation::allgather, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allgatherv, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allreduce, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoall, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoallv, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::bcast, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::barrier, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gather, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gatherv, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatter, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatterv, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatter, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatterv, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::send, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::recv, Al::HostTransferBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::sendrecv, Al::HostTransferBackend> : std::true_type {};

// Type support.
template <> struct IsTypeSupported<Al::HostTransferBackend, char> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, signed char> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, unsigned char> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, short> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, unsigned short> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, int> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, unsigned int> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, long> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, unsigned long> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, long long> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, unsigned long long> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, float> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, double> : std::true_type {};
template <> struct IsTypeSupported<Al::HostTransferBackend, long double> : std::true_type {};

// Reduction operator support (all are supported).
template <Al::ReductionOperator op>
struct IsReductionOpSupported<Al::HostTransferBackend, op> : std::true_type {};

// Backend name.
template <> constexpr char AlBackendName<Al::HostTransferBackend>[] = "ht";

// Algorithm types.
template <> struct OpAlgoType<AlOperation::allgather, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::allgather_algo_type;
};
template <> struct OpAlgoType<AlOperation::allgatherv, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::allgatherv_algo_type;
};
template <> struct OpAlgoType<AlOperation::allreduce, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::allreduce_algo_type;
};
template <> struct OpAlgoType<AlOperation::alltoall, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::alltoall_algo_type;
};
template <> struct OpAlgoType<AlOperation::alltoallv, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::alltoallv_algo_type;
};
template <> struct OpAlgoType<AlOperation::barrier, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::barrier_algo_type;
};
template <> struct OpAlgoType<AlOperation::bcast, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::bcast_algo_type;
};
template <> struct OpAlgoType<AlOperation::gather, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::gather_algo_type;
};
template <> struct OpAlgoType<AlOperation::gatherv, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::gatherv_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::reduce_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce_scatter, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::reduce_scatter_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce_scatterv, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::reduce_scatterv_algo_type;
};
template <> struct OpAlgoType<AlOperation::scatter, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::scatter_algo_type;
};
template <> struct OpAlgoType<AlOperation::scatterv, Al::HostTransferBackend> {
  using type = Al::HostTransferBackend::scatterv_algo_type;
};

// Supported algorithms.
template <>
std::vector<std::pair<std::string, typename OpAlgoType<AlOperation::allreduce, Al::HostTransferBackend>::type>> get_supported_algos<AlOperation::allreduce, Al::HostTransferBackend>() {
  using algo_type = Al::HostTransferBackend::allreduce_algo_type;
  return {{"automatic", algo_type::automatic},
          {"host_transfer", algo_type::host_transfer}};
}

// Algorithms.
template <>
struct AlgorithmOptions<Al::HostTransferBackend> {
  typename Al::HostTransferBackend::allgather_algo_type allgather_algo =
    Al::HostTransferBackend::allgather_algo_type::automatic;
  typename Al::HostTransferBackend::allgatherv_algo_type allgatherv_algo =
    Al::HostTransferBackend::allgatherv_algo_type::automatic;
  typename Al::HostTransferBackend::allreduce_algo_type allreduce_algo =
    Al::HostTransferBackend::allreduce_algo_type::automatic;
  typename Al::HostTransferBackend::alltoall_algo_type alltoall_algo =
    Al::HostTransferBackend::alltoall_algo_type::automatic;
  typename Al::HostTransferBackend::alltoallv_algo_type alltoallv_algo =
    Al::HostTransferBackend::alltoallv_algo_type::automatic;
  typename Al::HostTransferBackend::barrier_algo_type barrier_algo =
    Al::HostTransferBackend::barrier_algo_type::automatic;
  typename Al::HostTransferBackend::bcast_algo_type bcast_algo =
    Al::HostTransferBackend::bcast_algo_type::automatic;
  typename Al::HostTransferBackend::gather_algo_type gather_algo =
    Al::HostTransferBackend::gather_algo_type::automatic;
  typename Al::HostTransferBackend::gatherv_algo_type gatherv_algo =
    Al::HostTransferBackend::gatherv_algo_type::automatic;
  typename Al::HostTransferBackend::reduce_algo_type reduce_algo =
    Al::HostTransferBackend::reduce_algo_type::automatic;
  typename Al::HostTransferBackend::reduce_scatter_algo_type reduce_scatter_algo =
    Al::HostTransferBackend::reduce_scatter_algo_type::automatic;
  typename Al::HostTransferBackend::reduce_scatterv_algo_type reduce_scatterv_algo =
    Al::HostTransferBackend::reduce_scatterv_algo_type::automatic;
  typename Al::HostTransferBackend::scatter_algo_type scatter_algo =
    Al::HostTransferBackend::scatter_algo_type::automatic;
  typename Al::HostTransferBackend::scatterv_algo_type scatterv_algo =
    Al::HostTransferBackend::scatterv_algo_type::automatic;
};
