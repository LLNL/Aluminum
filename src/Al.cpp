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

#include "Al.hpp"
#include "internal.hpp"
#include "progress.hpp"
#ifdef AL_HAS_CUDA
#include "cuda.hpp"
#endif

namespace Al {

namespace {
// Whether the library has been initialized.
bool is_initialized = false;
// Progress engine.
internal::ProgressEngine* progress_engine = nullptr;
}

void Initialize(int& argc, char**& argv) {
  // Avoid repeated initialization.
  if (is_initialized) {
    return;
  }
  internal::mpi::init(argc, argv);
  progress_engine = new internal::ProgressEngine();
  progress_engine->run();
  is_initialized = true;
#ifdef AL_HAS_CUDA
  internal::cuda::init(argc, argv);
#endif
#ifdef AL_HAS_NCCL
  internal::nccl::init(argc, argv);
#endif
#ifdef AL_HAS_MPI_CUDA
  internal::mpi_cuda::init(argc, argv);
#endif
}

void Finalize() {
  // Make calling Finalize multiple times safely.
  if (!is_initialized) {
    return;
  }
  // Finalize in reverse order of initialization.
#ifdef AL_HAS_MPI_CUDA
  internal::mpi_cuda::finalize();
#endif
#ifdef AL_HAS_NCCL
  internal::nccl::finalize();
#endif
#ifdef AL_HAS_CUDA
  internal::cuda::finalize();
#endif
  progress_engine->stop();
  delete progress_engine;
  progress_engine = nullptr;
  is_initialized = false;
  internal::mpi::finalize();
}

bool Initialized() {
  return is_initialized;
}

namespace internal {

// Note: This is declared in progress.hpp.
ProgressEngine* get_progress_engine() {
  return progress_engine;
}

}  // namespace internal
}  // namespace Al
