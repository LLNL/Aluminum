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

#include <cstdlib>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "mpi_cuda/coll_util.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

void gpu_reduce_to_host(const std::vector<int> &gpus,
                        float *host_buf,
                        const std::vector<float *> &gbufs,
                        size_t count,
                        std::vector<cudaStream_t> &streams);

void gpu_reduce_to_host_pipeline(const std::vector<int> &gpus,
                                 float *host_buf,
                                 const std::vector<float *> &gpu_bufs,
                                 size_t count,
                                 std::vector<cudaStream_t> &streams1,
                                 std::vector<cudaStream_t> &streams2);

void gpu_allreduce(const std::vector<int> &gpus,
                   const std::vector<float *> &gbufs,
                   size_t count,
                   std::vector<cudaStream_t> &streams);

void gpu_allreduce2(const std::vector<int> &gpus,
                    const std::vector<float *> &gpu_data,
                    size_t count,
                    std::vector<cudaStream_t> &streams);


void gpu_broadcast(const std::vector<int> &gpus,
                   float *host_buf,                   
                   const std::vector<float *> &gbufs,
                   size_t count,
                   std::vector<cudaStream_t> &streams);

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
