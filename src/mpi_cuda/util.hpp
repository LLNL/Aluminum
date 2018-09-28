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

#include <cuda_runtime.h>
#include <mpi.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <cstdlib>

#define COLL_ASSERT(x) do {                                      \
    int s = x;                                                   \
    if (!s) {                                                    \
      std::cerr << "Assertion failure at "                       \
                << __FILE__ << ":" << __LINE__ << std::endl;     \
      abort();                                                   \
    }                                                            \
  } while (0)

#ifdef COLL_DEBUG
#define COLL_LOG_DEBUG(x) x;
#else
#define COLL_LOG_DEBUG(x)
#endif


#define COLL_CHECK_CUDA(cuda_call)                                      \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define COLL_CHECK_MPI(call)                                            \
  do {                                                                  \
    int status = call;                                                  \
    if (status != MPI_SUCCESS) {                                        \
      std::cerr << "MPI error" << std::endl;                            \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, status);                                \
    }                                                                   \
  } while (0)


namespace Al {
namespace internal {
namespace mpi_cuda {

inline int dec(int x, int t) {
  return (x -1 + t) % t;
}

inline int inc(int x, int t) {
  return (x + 1) % t;
}

inline void *malloc_aligned(size_t s) {
#if 1
  unsigned long align_size = sysconf(_SC_PAGESIZE);
  //std::cerr << "page size: " << align_size << std::endl;
  void *p = nullptr;
  if (posix_memalign(&p, align_size, s)) {
    std::cerr << "posix_memalign failed\n";
    abort();
  }
  return p;
#else
  return malloc(s);
#endif
}

inline
void create_streams(std::vector<cudaStream_t> &streams,
                    std::vector<int> &gpus) {
  COLL_ASSERT(streams.empty());
  int num_gpus = gpus.size();
  for (int i = 0; i < num_gpus; ++i) {
    COLL_CHECK_CUDA(cudaSetDevice(gpus[i]));
    cudaStream_t s;
    COLL_CHECK_CUDA(cudaStreamCreate(&s));
    streams.push_back(s);
  }
}

inline
void destroy_streams(std::vector<cudaStream_t> &streams,
                     std::vector<int> &gpus) {
  int num_gpus = gpus.size();
  for (int i = 0; i < num_gpus; ++i) {
    COLL_CHECK_CUDA(cudaSetDevice(gpus[i]));
    COLL_CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    COLL_CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }
  streams.clear();
}

template <typename T>
MPI_Datatype get_mpi_data_type() {
  return mpi::TypeMap<T>();
}

class MPIPrintStream {
 public:
  MPIPrintStream(std::ostream &os, int rank): m_os(os) {
    ss << "[" << rank << "] ";
  }
  ~MPIPrintStream() {
    m_os << ss.str();
  }
  std::stringstream &operator()() {
    return ss;
  }

 protected:
  std::ostream &m_os;
  std::stringstream ss;
};

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
