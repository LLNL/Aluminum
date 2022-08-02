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

/** Common simple utilities. */

#include <iostream>
#include <cstdlib>

#include <Al.hpp>


#ifdef AL_HAS_CUDA

/**
 * Return the number of GPUs to use on the system.
 *
 * By default this will use CUDA to determine how many GPUs there are.
 * This can be overridden using the AL_NUM_GPUS environment variable.
 */
inline int get_number_of_gpus() {
  int num_gpus = 0;
  char* env = std::getenv("AL_NUM_GPUS");
  if (env) {
    num_gpus = std::atoi(env);
    if (num_gpus == 0) {
      std::cerr << "AL_NUM_GPUS either 0 or invalid value: "
                << env << std::endl;
      std::abort();
    }
  } else {
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaGetDeviceCount(&num_gpus));
  }
  return num_gpus;
}

#endif  /** AL_HAS_CUDA */

/** Attempt to identify the local rank on a node from the environment. */
inline int get_local_rank() {
  char* env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) {
    env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  }
  if (!env) {
    env = std::getenv("SLURM_LOCALID");
  }
  if (!env) {
    std::cerr << "Cannot determine local rank" << std::endl;
    std::abort();
  }
  return std::atoi(env);
}

/** Attempt to identify the number of ranks on a node from the environment. */
inline int get_local_size() {
  char *env = std::getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env) {
    env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  }
  if (!env) {
    env = std::getenv("SLURM_NTASKS_PER_NODE");
  }
  if (!env) {
    std::cerr << "Cannot determine local size" << std::endl;
    std::abort();
  }
  return std::atoi(env);
}
