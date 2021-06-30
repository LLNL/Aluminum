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
#include <cstdint>
#include <cassert>
#include "test_utils.hpp"

/** Emulate most std::vector functionality but with CUDA memory. */
template <typename T>
class CUDAVector {
 public:
  CUDAVector() : m_count(0), m_ptr(nullptr) {}

  CUDAVector(size_t count, cudaStream_t stream = 0) : m_count(count), m_ptr(nullptr),
                                                      m_stream(stream) {
    allocate();
  }

  CUDAVector(const std::vector<T> &host_vector, cudaStream_t stream = 0) :
    m_count(host_vector.size()), m_ptr(nullptr), m_stream(stream) {
    allocate();
    sync_memcpy(m_ptr, host_vector.data(), get_bytes(), cudaMemcpyDefault);
  }

  CUDAVector(const CUDAVector &v) : m_count(v.m_count), m_ptr(nullptr),
                                    m_stream(v.m_stream) {
    allocate();
    sync_memcpy(m_ptr, v.data(), get_bytes(), cudaMemcpyDefault);
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
      Al::internal::mempool.release<Al::internal::MemoryType::CUDA>(m_ptr);
      m_ptr = nullptr;
      m_count = 0;
    }
  }

  void allocate() {
    assert(m_ptr == nullptr);
    if (m_count > 0) {
      m_ptr = Al::internal::mempool.allocate<Al::internal::MemoryType::CUDA, T>(
        m_count, m_stream);
    }
  }

  CUDAVector &operator=(const CUDAVector<T> &v) {
    if (size() != v.size()) {
      clear();
      m_count = v.m_count;
      allocate();
    }
    sync_memcpy(m_ptr, v.m_ptr, get_bytes(), cudaMemcpyDefault);
    return *this;
  }

  CUDAVector& move(const CUDAVector<T> &v) {
    sync_memcpy(m_ptr, v.m_ptr, v.get_bytes(), cudaMemcpyDefault);
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
    sync_memcpy(hv.data(), m_ptr, get_bytes(), cudaMemcpyDeviceToHost);
    return hv;
  }

  void copyout(std::vector<T>& hv) const {
    sync_memcpy(hv.data(), m_ptr, get_bytes(), cudaMemcpyDeviceToHost);
  }

  void copyin(const T *hp) {
    sync_memcpy(m_ptr, hp, get_bytes(), cudaMemcpyHostToDevice);
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
                   cudaMemcpyKind kind) const {
    if (count == 0) {
      return;
    }
    AL_FORCE_CHECK_CUDA_NOSYNC(
      cudaMemcpyAsync(dst, src, count, kind, m_stream));
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaStreamSynchronize(m_stream));
  }

 private:
  size_t m_count;
  T *m_ptr;
  cudaStream_t m_stream;
};

/** Compare two vectors with a given tolerance. */
template <typename T>
bool check_vector(const std::vector<T>& expected,
                  const CUDAVector<T>& actual,
                  size_t start = 0,
                  size_t end = std::numeric_limits<size_t>::max(),
                  const T eps = T(1e-4)) {
  std::vector<T>&& actual_host = actual.copyout();
  return check_vector(expected, actual_host, start, end, eps);
}
