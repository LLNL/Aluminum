#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "test_utils.hpp"

#define NCCL_THRESHOLD	1e-05

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
    cudaMemcpy(m_ptr, host_vector.data(), get_bytes(), cudaMemcpyDefault);
  }

  CUDAVector(const CUDAVector &v): m_count(v.m_count), m_ptr(nullptr) {
    allocate();
    cudaMemcpy(m_ptr, v.data(), get_bytes(), cudaMemcpyDefault);
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
      cudaFree(m_ptr);
      m_ptr = nullptr;
      m_count = 0;
    }
  }

  void allocate() {
    if (m_count > 0) {
      cudaMalloc(&m_ptr, get_bytes());
    }
  }

  CUDAVector &operator=(const CUDAVector<T> &v) {
    clear();
    m_count = v.m_count;
    allocate();
    cudaMemcpy(m_ptr, v.m_ptr, get_bytes(), cudaMemcpyDefault);
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
    cudaMemcpy(hv.data(), m_ptr, get_bytes(), cudaMemcpyDeviceToHost);
    return hv;
  }
  
  void copyin(const T *hp) {
    cudaMemcpy(m_ptr, hp, get_bytes(), cudaMemcpyHostToDevice);
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

template <>
struct VectorType<allreduces::NCCLBackend> {
  using type = CUDAVector<float>;
};

template <>
struct VectorType<allreduces::MPICUDABackend> {
  using type = CUDAVector<float>;
};

template <>
typename VectorType<allreduces::NCCLBackend>::type
gen_data<allreduces::NCCLBackend>(size_t count) {
  auto &&host_data = gen_data<allreduces::MPIBackend>(count);
  CUDAVector<float> data(host_data);
  return data;
}

template <>
typename VectorType<allreduces::MPICUDABackend>::type
gen_data<allreduces::MPICUDABackend>(size_t count) {
  auto &&host_data = gen_data<allreduces::MPIBackend>(count);
  CUDAVector<float> data(host_data);
  return data;
}

bool check_vector(const CUDAVector<float>& expected,
                  const CUDAVector<float>& actual) {
  std::vector<float> &&expected_host = expected.copyout();
  std::vector<float> &&actual_host= actual.copyout();
  return check_vector(expected_host, actual_host);
}

void get_expected_result(CUDAVector<float>& expected) {
  std::vector<float> &&host_data = expected.copyout();
  MPI_Allreduce(MPI_IN_PLACE, host_data.data(), expected.size(),
                MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  expected.copyin(host_data);
}


template <>
std::vector<typename allreduces::NCCLBackend::algo_type>
get_nb_allreduce_algorithms<allreduces::NCCLBackend>() {
  // NCCLBackend does not have non-blocking interface implemented
  std::vector<typename allreduces::NCCLBackend::algo_type> algos = {};
  return algos;
}

template <>
std::vector<typename allreduces::MPICUDABackend::algo_type>
get_allreduce_algorithms<allreduces::MPICUDABackend>() {
  std::vector<typename allreduces::MPICUDABackend::algo_type> algos = {
    allreduces::MPICUDABackend::algo_type::automatic,
    allreduces::MPICUDABackend::algo_type::ring,
    allreduces::MPICUDABackend::algo_type::bi_ring
  };
  return algos;
}

template <>
std::vector<typename allreduces::MPICUDABackend::algo_type>
get_nb_allreduce_algorithms<allreduces::MPICUDABackend>() {
  std::vector<typename allreduces::MPICUDABackend::algo_type> algos = {
    allreduces::MPICUDABackend::algo_type::automatic,
    allreduces::MPICUDABackend::algo_type::ring,
    allreduces::MPICUDABackend::algo_type::bi_ring
  };
  return algos;
}
