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

#include <memory>
#include <utility>
#include <assert.h>
#include <cuda_runtime.h>
#include "test_utils.hpp"

#define NCCL_THRESHOLD	1e-05

// For macros.
using al_exception = Al::al_exception;

int get_number_of_gpus() {
  int num_gpus = 0;
  char *env = getenv("ALUMINUM_NUM_GPUS");
  if (env) {
    std::cout << "Number of GPUs set by ALUMINUM_NUM_GPUS\n";
    num_gpus = atoi(env);
  } else {
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaGetDeviceCount(&num_gpus));    
  }
  return num_gpus;
}

int get_local_rank() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("SLURM_LOCALID");
  if (!env) {
    std::cerr << "Can't determine local rank\n";
    abort();
  }
  return atoi(env);
}

int get_local_size() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = getenv("SLURM_NTASKS_PER_NODE");
  if (!env) {
    std::cerr << "Can't determine local size\n";
    abort();
  }
  return atoi(env);
}

inline int set_device() {
  int num_gpus = get_number_of_gpus();
  int local_rank = get_local_rank();
  int local_size = get_local_size();
  if (num_gpus < local_size) {
    std::cerr << "Number of available GPUs, " << num_gpus
              << ", is smaller than the number of local MPI ranks, "
              << local_size << "\n";
    abort();
  }    
  int device = local_rank;
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaSetDevice(device));
  return device;
}

template <typename T>
class CUDAVector {
 public:
  CUDAVector(): m_count(0), m_ptr(nullptr) {}
    
  CUDAVector(size_t count): m_count(count), m_ptr(nullptr) {
    allocate();
  }

  CUDAVector(const std::vector<T> &host_vector):
      m_count(host_vector.size()), m_ptr(nullptr) {
    allocate();
    AL_FORCE_CHECK_CUDA(cudaMemcpy(m_ptr, host_vector.data(), get_bytes(), cudaMemcpyDefault));
  }

  CUDAVector(const CUDAVector &v): m_count(v.m_count), m_ptr(nullptr) {
    allocate();
    AL_FORCE_CHECK_CUDA(cudaMemcpy(m_ptr, v.data(), get_bytes(), cudaMemcpyDefault));
  }

  CUDAVector(CUDAVector &&v): CUDAVector() {
    swap(*this, v);
  }
  
  ~CUDAVector() {
    clear();
  }

  friend void swap(CUDAVector &x, CUDAVector &y) {
    using std::swap;
    swap(x.m_count, y.m_count);
    swap(x.m_ptr, y.m_ptr);
  }

  size_t size() const {
    return m_count;
  }

  size_t get_bytes() const {
    return m_count * sizeof(T);
  }

  void clear() {
    if (m_count > 0) {
      AL_FORCE_CHECK_CUDA(cudaFree(m_ptr));
      m_ptr = nullptr;
      m_count = 0;
    }
  }

  void allocate() {
    assert(m_ptr == nullptr);
    if (m_count > 0) {
#if 0
      AL_FORCE_CHECK_CUDA(cudaMalloc(&m_ptr, get_bytes()));
#else
      cudaError_t e = cudaMalloc(&m_ptr, get_bytes());
      if (e != cudaSuccess) {
        
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cerr << "Error allocating "
                  << get_bytes() << " bytes of memory: "
                  << cudaGetErrorString(e)
                  << ", free: " << free_mem << "\n";
        cudaDeviceReset();
        abort();
      }
#endif
    }
  }

  CUDAVector &operator=(const CUDAVector<T> &v) {
    if (size() != v.size()) {
      clear();
      m_count = v.m_count;
      allocate();
    }
    AL_FORCE_CHECK_CUDA(cudaMemcpy(m_ptr, v.m_ptr, get_bytes(), cudaMemcpyDefault));
    return *this;
  }

  CUDAVector& move(const CUDAVector<T> &v) {
    AL_FORCE_CHECK_CUDA(cudaMemcpy(m_ptr, v.m_ptr, v.get_bytes(), cudaMemcpyDefault));
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
    AL_FORCE_CHECK_CUDA(cudaMemcpy(hv.data(), m_ptr, get_bytes(), cudaMemcpyDeviceToHost));
    return hv;
  }

  void copyout(std::vector<T>& hv) const {
    AL_FORCE_CHECK_CUDA(cudaMemcpy(hv.data(), m_ptr, get_bytes(), cudaMemcpyDeviceToHost));
  }
  
  void copyin(const T *hp) {
    AL_FORCE_CHECK_CUDA(cudaMemcpy(m_ptr, hp, get_bytes(), cudaMemcpyHostToDevice));
  }

  void copyin(const std::vector<T> &hv) {
    clear();
    m_count = hv.size();
    allocate();
    copyin(hv.data());
  }
  
 private:
  size_t m_count;  
  T *m_ptr;
};


bool check_vector(const CUDAVector<float>& expected,
                  const CUDAVector<float>& actual) {
  std::vector<float> &&expected_host = expected.copyout();
  std::vector<float> &&actual_host = actual.copyout();
  return check_vector(expected_host, actual_host);
}

void get_expected_result(CUDAVector<float>& expected) {
  std::vector<float> &&host_data = expected.copyout();
  MPI_Allreduce(MPI_IN_PLACE, host_data.data(), expected.size(),
                MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  expected.copyin(host_data);
}

inline std::pair<cudaEvent_t, cudaEvent_t> get_timer_events() {
  static bool inited = false;
  static cudaEvent_t start;
  static cudaEvent_t end;
  if (!inited) {
    AL_FORCE_CHECK_CUDA(cudaEventCreate(&start));
    AL_FORCE_CHECK_CUDA(cudaEventCreate(&end));
    inited = true;
  }
  return std::make_pair(start, end);
}
