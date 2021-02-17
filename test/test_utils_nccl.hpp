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
struct VectorType<T, Al::NCCLBackend> {
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

// Specialization for half. Standard RNGs do not support half.
template <> struct VectorType<__half, Al::NCCLBackend> {
  using type = CUDAVector<__half>;

  static type gen_data(size_t count) {
    auto&& host_data = VectorType<float, Al::MPIBackend>::gen_data(count);
    std::vector<__half> host_data_half(count);
    for (size_t i = 0; i < count; ++i) {
      host_data_half[i] = __float2half(host_data[i]);
    }
    CUDAVector<__half> data(host_data_half);
    return data;
  }

  static std::vector<__half> copy_to_host(const type& v) {
    return v.copyout();
  }
};

// Specialize to create a CUDA stream with the communicator.
template <>
CommWrapper<Al::NCCLBackend>::CommWrapper(MPI_Comm mpi_comm) {
  cudaStream_t stream;
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamCreate(&stream));
  comm_ = std::make_unique<typename Al::NCCLBackend::comm_type>(
    mpi_comm, stream);
}
template <>
CommWrapper<Al::NCCLBackend>::~CommWrapper() noexcept(false) {
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamDestroy(comm_->get_stream()));
}

template <>
void complete_operations<Al::NCCLBackend>(
  typename Al::NCCLBackend::comm_type& comm) {
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamSynchronize(comm.get_stream()));
}


// Operator support.
template <> struct IsOpSupported<AlOperation::allgather, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allgatherv, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::allreduce, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoall, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::alltoallv, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::barrier, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::bcast, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gather, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::gatherv, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatter, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::reduce_scatterv, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatter, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::scatterv, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::send, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::recv, Al::NCCLBackend> : std::true_type {};
template <> struct IsOpSupported<AlOperation::sendrecv, Al::NCCLBackend> : std::true_type {};

// Type support.
template <> struct IsTypeSupported<Al::NCCLBackend, char> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, unsigned char> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, int> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, unsigned int> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, long long int> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, unsigned long long int> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, __half> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, float> : std::true_type {};
template <> struct IsTypeSupported<Al::NCCLBackend, double> : std::true_type {};

// Reduction operator support.
template <>
struct IsReductionOpSupported<Al::NCCLBackend, Al::ReductionOperator::sum> : std::true_type {};
template <>
struct IsReductionOpSupported<Al::NCCLBackend, Al::ReductionOperator::prod> : std::true_type {};
template <>
struct IsReductionOpSupported<Al::NCCLBackend, Al::ReductionOperator::min> : std::true_type {};
template <>
struct IsReductionOpSupported<Al::NCCLBackend, Al::ReductionOperator::max> : std::true_type {};

// Backend name.
template <> constexpr char AlBackendName<Al::NCCLBackend>[] = "nccl";

// Algorithm types.
template <> struct OpAlgoType<AlOperation::allgather, Al::NCCLBackend> {
  using type = Al::NCCLBackend::allgather_algo_type;
};
template <> struct OpAlgoType<AlOperation::allgatherv, Al::NCCLBackend> {
  using type = Al::NCCLBackend::allgatherv_algo_type;
};
template <> struct OpAlgoType<AlOperation::allreduce, Al::NCCLBackend> {
  using type = Al::NCCLBackend::allreduce_algo_type;
};
template <> struct OpAlgoType<AlOperation::alltoall, Al::NCCLBackend> {
  using type = Al::NCCLBackend::alltoall_algo_type;
};
template <> struct OpAlgoType<AlOperation::alltoallv, Al::NCCLBackend> {
  using type = Al::NCCLBackend::alltoallv_algo_type;
};
template <> struct OpAlgoType<AlOperation::barrier, Al::NCCLBackend> {
  using type = Al::NCCLBackend::barrier_algo_type;
};
template <> struct OpAlgoType<AlOperation::bcast, Al::NCCLBackend> {
  using type = Al::NCCLBackend::bcast_algo_type;
};
template <> struct OpAlgoType<AlOperation::gather, Al::NCCLBackend> {
  using type = Al::NCCLBackend::gather_algo_type;
};
template <> struct OpAlgoType<AlOperation::gatherv, Al::NCCLBackend> {
  using type = Al::NCCLBackend::gatherv_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce, Al::NCCLBackend> {
  using type = Al::NCCLBackend::reduce_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce_scatter, Al::NCCLBackend> {
  using type = Al::NCCLBackend::reduce_scatter_algo_type;
};
template <> struct OpAlgoType<AlOperation::reduce_scatterv, Al::NCCLBackend> {
  using type = Al::NCCLBackend::reduce_scatterv_algo_type;
};
template <> struct OpAlgoType<AlOperation::scatter, Al::NCCLBackend> {
  using type = Al::NCCLBackend::scatter_algo_type;
};
template <> struct OpAlgoType<AlOperation::scatterv, Al::NCCLBackend> {
  using type = Al::NCCLBackend::scatterv_algo_type;
};

// Algorithms.
template <>
struct AlgorithmOptions<Al::NCCLBackend> {
  typename Al::NCCLBackend::allgather_algo_type allgather_algo =
    Al::NCCLBackend::allgather_algo_type::automatic;
  typename Al::NCCLBackend::allgatherv_algo_type allgatherv_algo =
    Al::NCCLBackend::allgatherv_algo_type::automatic;
  typename Al::NCCLBackend::allreduce_algo_type allreduce_algo =
    Al::NCCLBackend::allreduce_algo_type::automatic;
  typename Al::NCCLBackend::alltoall_algo_type alltoall_algo =
    Al::NCCLBackend::alltoall_algo_type::automatic;
  typename Al::NCCLBackend::alltoallv_algo_type alltoallv_algo =
    Al::NCCLBackend::alltoallv_algo_type::automatic;
  typename Al::NCCLBackend::barrier_algo_type barrier_algo =
    Al::NCCLBackend::barrier_algo_type::automatic;
  typename Al::NCCLBackend::bcast_algo_type bcast_algo =
    Al::NCCLBackend::bcast_algo_type::automatic;
  typename Al::NCCLBackend::gather_algo_type gather_algo =
    Al::NCCLBackend::gather_algo_type::automatic;
  typename Al::NCCLBackend::gatherv_algo_type gatherv_algo =
    Al::NCCLBackend::gatherv_algo_type::automatic;
  typename Al::NCCLBackend::reduce_algo_type reduce_algo =
    Al::NCCLBackend::reduce_algo_type::automatic;
  typename Al::NCCLBackend::reduce_scatter_algo_type reduce_scatter_algo =
    Al::NCCLBackend::reduce_scatter_algo_type::automatic;
  typename Al::NCCLBackend::reduce_scatterv_algo_type reduce_scatterv_algo =
    Al::NCCLBackend::reduce_scatterv_algo_type::automatic;
  typename Al::NCCLBackend::scatter_algo_type scatter_algo =
    Al::NCCLBackend::scatter_algo_type::automatic;
  typename Al::NCCLBackend::scatterv_algo_type scatterv_algo =
    Al::NCCLBackend::scatterv_algo_type::automatic;
};
