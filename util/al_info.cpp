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

#include <Al_config.hpp>
#include <aluminum/datatypes.hpp>

#include <iostream>


int main(int, char**) {
  std::cout << "Aluminum "
            << AL_VERSION
            << " (" << AL_GIT_VERSION << ")\n";

  std::cout << "Backends:";
  std::cout << " mpi";  // MPI is always present.
#ifdef AL_HAS_NCCL
  std::cout << " nccl";
#endif
#ifdef AL_HAS_HOST_TRANSFER
  std::cout << " ht";
#endif
#ifdef AL_HAS_MPI_CUDA
  std::cout << " mpi-cuda";
#endif
  std::cout << "\n";
  std::cout << "Features:";
#ifdef AL_DEBUG
  std::cout << " debug";
#endif
#ifdef AL_THREAD_MULTIPLE
  std::cout << " thread-multiple";
#endif
#ifdef AL_HAS_CUDA
  std::cout << " cuda";
#endif
#ifdef AL_HAS_ROCM
  std::cout << " rocm";
#endif
#ifdef AL_HAS_MPI_CUDA_RMA
  std::cout << " mpi-cuda-rma";
#endif
#ifdef AL_DEBUG_HANG_CHECK
  std::cout << " hang-check";
#endif
#ifdef AL_HAS_PROF
  std::cout << " prof";
#endif
#ifdef AL_HAS_NVPROF
  std::cout << " nvprof";
#endif
#ifdef AL_HAS_ROCTRACER
  std::cout << " roctracer";
#endif
#ifdef AL_TRACE
  std::cout << " trace";
#endif
#ifdef AL_MPI_SERIALIZE
  std::cout << " mpi-serialize";
#endif
#ifdef AL_HAS_HALF
  std::cout << " half";
#endif
#ifdef AL_HAS_BFLOAT
  std::cout << " bfloat";
#endif
  std::cout << std::endl;
  return 0;
}
