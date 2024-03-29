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
#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>
#include <cstdint>
#include <cassert>

#include <Al.hpp>


#if defined AL_HAS_CUDA || defined AL_HAS_ROCM

#if !defined AlGpuMalloc
# define AlGpuMalloc AL_GPU_RT(Malloc)
#endif

#if !defined AlGpuFree
# define AlGpuFree AL_GPU_RT(Free)
#endif

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
    AL_FORCE_CHECK_GPU_NOSYNC(AlGpuGetDeviceCount(&num_gpus));
  }
  return num_gpus;
}

#endif  /** AL_HAS_CUDA || AL_HAS_ROCM */

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
    env = std::getenv("FLUX_TASK_LOCAL_ID");
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
  // Flux doesn't have an environment variable for this directly, so we
  // assume an even distribution.
  if (!env) {
    char* flux_size = std::getenv("FLUX_JOB_SIZE");
    if (flux_size) {
      char* flux_nnodes = std::getenv("FLUX_JOB_NNODES");
      if (flux_nnodes) {
        int size = std::atoi(flux_size);
        int nnodes = std::atoi(flux_nnodes);
        return (size + nnodes - 1) / nnodes;
      }
    }
  }
  if (!env) {
    std::cerr << "Cannot determine local size" << std::endl;
    std::abort();
  }
  return std::atoi(env);
}


// Utilities for managing arrays of data that exist on the appropriate
// device for a backend (i.e., CPU for MPI, GPU for NCCL/HostTransfer).

/** Generate a random floating point value. */
template <typename T, typename Generator,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
T gen_random_val(Generator& g) {
  std::uniform_real_distribution<T> rng;
  return rng(g);
}

/** Generate a random integral value. */
template <typename T, typename Generator,
          std::enable_if_t<std::is_integral<T>::value, bool> = true>
T gen_random_val(Generator& g) {
  std::uniform_int_distribution<T> rng;
  return rng(g);
}

/** Helper for generating random vectors. */
template <typename T>
struct RandVectorGen {
  template <typename Generator>
  static std::vector<T> gen(size_t count, Generator& g) {
    std::vector<T> v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = gen_random_val<T>(g);
    }
    return v;
  }
};
#ifdef AL_HAS_HALF
// Specialization for half. Standard RNGs do not support half.
template <>
struct RandVectorGen<__half> {
  template <typename Generator>
  static std::vector<__half> gen(size_t count, Generator& g) {
    std::vector<__half> v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = __float2half(gen_random_val<float>(g));
    }
    return v;
  }
};
#endif
#ifdef AL_HAS_BFLOAT
// Specialization for bfloat. Standard RNGs do not support bfloat.
template <>
struct RandVectorGen<al_bfloat16> {
  template <typename Generator>
  static std::vector<al_bfloat16> gen(size_t count, Generator& g) {
    std::vector<al_bfloat16> v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = __float2bfloat16(gen_random_val<float>(g));
    }
    return v;
  }
};
#endif

/**
 * Identify a vector type for each backend and support generating an
 * instance of it with random data.
 *
 * This is std::vector<T> by default.
 */
template <typename T, typename Backend>
struct VectorType {
  using type = std::vector<T>;

  /** Generate a vector of random data of size count. */
  static type gen_data(size_t count, int = 0) {
    static bool rng_seeded = false;
    static std::minstd_rand rng_gen;
    if (!rng_seeded) {
      // Seed using the MPI rank (only if MPI has been initialized).
      int flag;
      MPI_Initialized(&flag);
      if (flag) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rng_gen.seed(rank + 1);
        rng_seeded = true;
      }
    }
    return RandVectorGen<T>::gen(count, rng_gen);
  }

  /** Return a copy of the data on the host. */
  static std::vector<T> copy_to_host(const type& v) {
    return std::vector<T>(v);
  }
};

/** Return an uninitialized vector of size count. */
template <typename T, typename Backend>
typename VectorType<T, Backend>::type get_vector(size_t count) {
  return typename VectorType<T, Backend>::type(count);
}

#if defined AL_HAS_CUDA || defined AL_HAS_ROCM

// Note: This is adapted from the same class in the Aluminum test utils
// but does not use the Aluminum memory pool to simplify things.

/** Like an std::vector, but with CUDA memory. */
template <typename T>
class CUDAVector {
 public:
  CUDAVector() : m_count(0), m_ptr(nullptr) {}

  CUDAVector(size_t count, AlGpuStream_t stream = 0) :
    m_count(count), m_ptr(nullptr), m_stream(stream) {
    allocate();
  }

  CUDAVector(const std::vector<T> &host_vector, AlGpuStream_t stream = 0) :
    m_count(host_vector.size()), m_ptr(nullptr), m_stream(stream) {
    allocate();
    sync_memcpy(m_ptr, host_vector.data(), get_bytes(), AlGpuMemcpyDefault);
  }

  CUDAVector(const CUDAVector &v) : m_count(v.m_count), m_ptr(nullptr),
                                    m_stream(v.m_stream) {
    allocate();
    sync_memcpy(m_ptr, v.data(), get_bytes(), AlGpuMemcpyDefault);
  }

  CUDAVector(CUDAVector &&v) : CUDAVector() {
    swap(*this, v);
  }

  ~CUDAVector() {
    clear();
  }

  friend void swap(CUDAVector &x, CUDAVector &y) {
    using std::swap;
    swap(x.m_count, y.m_count);
    swap(x.m_ptr, y.m_ptr);
    swap(x.m_stream, y.m_stream);
  }

  size_t size() const {
    return m_count;
  }

  void clear() {
    if (m_count > 0) {
      AL_FORCE_CHECK_GPU(AlGpuFree(m_ptr));
      m_ptr = nullptr;
      m_count = 0;
    }
  }

  void allocate() {
    assert(m_ptr == nullptr);
    if (m_count > 0) {
      AL_FORCE_CHECK_GPU(AlGpuMalloc(&m_ptr, get_bytes()));
    }
  }

  CUDAVector &operator=(const CUDAVector<T> &v) {
    if (size() != v.size()) {
      clear();
      m_count = v.m_count;
      allocate();
    }
    sync_memcpy(m_ptr, v.m_ptr, get_bytes(), AlGpuMemcpyDefault);
    return *this;
  }

  CUDAVector& move(const CUDAVector<T> &v) {
    sync_memcpy(m_ptr, v.m_ptr, v.get_bytes(), AlGpuMemcpyDefault);
    return *this;
  }

  T *data() {
    return m_ptr;
  }

  const T *data() const {
    return m_ptr;
  }

  std::vector<T> copyout() const {
    std::vector<T> hv(size());
    sync_memcpy(hv.data(), m_ptr, get_bytes(), AlGpuMemcpyDeviceToHost);
    return hv;
  }

  void copyout(std::vector<T>& hv) const {
    sync_memcpy(hv.data(), m_ptr, get_bytes(), AlGpuMemcpyDeviceToHost);
  }

  void copyin(const T *hp) {
    sync_memcpy(m_ptr, hp, get_bytes(), AlGpuMemcpyHostToDevice);
  }

  void copyin(const std::vector<T> &hv) {
    clear();
    m_count = hv.size();
    allocate();
    copyin(hv.data());
  }

 protected:
  size_t get_bytes() const {
    return m_count * sizeof(T);
  }

  void sync_memcpy(void* dst, const void* src, size_t count,
                   AlGpuMemcpyKind kind) const {
    if (count == 0) {
      return;
    }
    AL_FORCE_CHECK_GPU_NOSYNC(
      AlGpuMemcpyAsync(dst, src, count, kind, m_stream));
    AL_FORCE_CHECK_GPU_NOSYNC(AlGpuStreamSynchronize(m_stream));
  }

 private:
  size_t m_count;
  T *m_ptr;
  AlGpuStream_t m_stream;
};

#endif  /** AL_HAS_CUDA || AL_HAS_ROCM */

// Specialize VectorType for different Aluminum backends.

#ifdef AL_HAS_NCCL

/** Version of VectorType for the NCCLBackend, using GPU data. */
template <typename T>
struct VectorType<T, Al::NCCLBackend> {
  using type = CUDAVector<T>;

  static type gen_data(size_t count, AlGpuStream_t stream = 0) {
    auto&& host_data = VectorType<T, Al::MPIBackend>::gen_data(count);
    CUDAVector<T> data(host_data, stream);
    return data;
  }

  static std::vector<T> copy_to_host(const type& v) {
    return v.copyout();
  }
};


#endif  /** AL_HAS_NCCL */

#ifdef AL_HAS_HOST_TRANSFER

/** Version of VectorType for the HostTransferBackend, using GPU data. */
template <typename T>
struct VectorType<T, Al::HostTransferBackend> {
  using type = CUDAVector<T>;

  static type gen_data(size_t count, AlGpuStream_t stream = 0) {
    auto&& host_data = VectorType<T, Al::MPIBackend>::gen_data(count);
    CUDAVector<T> data(host_data, stream);
    return data;
  }

  static std::vector<T> copy_to_host(const type& v) {
    return v.copyout();
  }
};

#endif  /** AL_HAS_HOST_TRANSFER */

// Utilities for ensuring operations complete.

/**
 * Ensure all enqueued Aluminum operations on a communicator complete.
 *
 * For backends that use compute streams on other devices (e.g., NCCL
 * on GPUs), this ensures the operations complete. For other backends,
 * this is a nop.
 */
template <typename Backend>
void complete_operations(typename Backend::comm_type &) {}

#ifdef AL_HAS_NCCL

template <>
void complete_operations<Al::NCCLBackend>(
  typename Al::NCCLBackend::comm_type& comm) {
  AL_FORCE_CHECK_GPU_NOSYNC(AlGpuStreamSynchronize(comm.get_stream()));
}

#endif  /** AL_HAS_NCCL */

#ifdef AL_HAS_HOST_TRANSFER

template <>
void complete_operations<Al::HostTransferBackend>(
  typename Al::HostTransferBackend::comm_type& comm) {
  AL_FORCE_CHECK_GPU_NOSYNC(AlGpuStreamSynchronize(comm.get_stream()));
}

#endif  /** AL_HAS_HOST_TRANSFER */
