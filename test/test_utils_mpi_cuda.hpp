#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "test_utils_cuda.hpp"

template <>
struct VectorType<allreduces::MPICUDABackend> {
  using type = CUDAVector<float>;
};

template <>
typename VectorType<allreduces::MPICUDABackend>::type
gen_data<allreduces::MPICUDABackend>(size_t count) {
  auto &&host_data = gen_data<allreduces::MPIBackend>(count);
  CUDAVector<float> data(host_data);
  return data;
}

template <>
std::vector<typename allreduces::MPICUDABackend::algo_type>
get_allreduce_algorithms<allreduces::MPICUDABackend>() {
  std::vector<typename allreduces::MPICUDABackend::algo_type> algos = {
    allreduces::MPICUDABackend::algo_type::ring,    
    allreduces::MPICUDABackend::algo_type::bi_ring
  };
  return algos;
}

template <>
std::vector<typename allreduces::MPICUDABackend::algo_type>
get_nb_allreduce_algorithms<allreduces::MPICUDABackend>() {
  std::vector<typename allreduces::MPICUDABackend::algo_type> algos = {
    allreduces::MPICUDABackend::algo_type::ring,
    allreduces::MPICUDABackend::algo_type::bi_ring
  };
  return algos;
}

template <>
inline typename allreduces::MPICUDABackend::req_type
get_request<allreduces::MPICUDABackend>() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  return s;
}
