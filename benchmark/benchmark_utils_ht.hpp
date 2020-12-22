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
#include "benchmark_utils.hpp"


template <>
struct Timer<Al::HostTransferBackend> {
  Timer() {
    AL_FORCE_CHECK_CUDA(cudaEventCreate(&start_event));
    AL_FORCE_CHECK_CUDA(cudaEventCreate(&end_event));
  }

  ~Timer() noexcept(false) {
    AL_FORCE_CHECK_CUDA(cudaEventDestroy(start_event));
    AL_FORCE_CHECK_CUDA(cudaEventDestroy(end_event));
  }

  void start_timer(typename Al::HostTransferBackend::comm_type& comm) {
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventRecord(start_event, comm.get_stream()));
  }

  double end_timer(typename Al::HostTransferBackend::comm_type &comm) {
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventRecord(end_event, comm.get_stream()));
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventSynchronize(end_event));
    float elapsed_time;
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventElapsedTime(
                                 &elapsed_time, start_event, end_event));
    // Convert milliseconds to seconds.
    return elapsed_time / 1000.0;
  }

  cudaEvent_t start_event;
  cudaEvent_t end_event;
};
