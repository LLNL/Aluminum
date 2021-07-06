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
struct VectorType<T, Al::MPICUDABackend> {
  using type = CUDAVector<T>;

  static type gen_data(size_t count, cudaStream_t stream = 0) {
    auto&& host_data = VectorType<T, Al::MPIBackend>::gen_data(count);
    CUDAVector<T> data(host_data, stream);
    return data;
  }

  static std::vector<T> copy_to_host(const type& v) {
    return v.copyout();
  }
};

// Specialize to use the Aluminum stream pool, and size it appropriately.
template <>
struct StreamManager<Al::MPICUDABackend> {
  using StreamType = cudaStream_t;

  static void init(size_t num_streams) {
    Al::internal::cuda::stream_pool.clear();
    Al::internal::cuda::stream_pool.allocate(num_streams);
  }
  static void finalize() {}
  static StreamType get_stream() {
    return Al::internal::cuda::stream_pool.get_stream();
  }
};

// Specialize to create a CUDA stream with the communicator.
template <>
CommWrapper<Al::MPICUDABackend>::CommWrapper(MPI_Comm mpi_comm) {
  comm_ = std::make_unique<typename Al::MPICUDABackend::comm_type>(
    mpi_comm, StreamManager<Al::MPICUDABackend>::get_stream());
}

template <>
void complete_operations<Al::MPICUDABackend>(
  typename Al::MPICUDABackend::comm_type& comm) {
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamSynchronize(comm.get_stream()));
}
