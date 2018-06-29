#pragma once

#include <memory>
#include <cuda_runtime.h>
#include "test_utils_cuda.hpp"

template <>
struct VectorType<Al::MPICUDABackend> {
  using type = CUDAVector<float>;
};

template <>
typename VectorType<Al::MPICUDABackend>::type
gen_data<Al::MPICUDABackend>(size_t count) {
  auto &&host_data = gen_data<Al::MPIBackend>(count);
  CUDAVector<float> data(host_data);
  return data;
}

template <>
std::vector<typename Al::MPICUDABackend::algo_type>
get_allreduce_algorithms<Al::MPICUDABackend>() {
  std::vector<typename Al::MPICUDABackend::algo_type> algos = {
    Al::MPICUDABackend::algo_type::ring,    
    Al::MPICUDABackend::algo_type::bi_ring
  };
  return algos;
}

template <>
std::vector<typename Al::MPICUDABackend::algo_type>
get_nb_allreduce_algorithms<Al::MPICUDABackend>() {
  std::vector<typename Al::MPICUDABackend::algo_type> algos = {
    Al::MPICUDABackend::algo_type::ring,
    Al::MPICUDABackend::algo_type::bi_ring
  };
  return algos;
}

template <>
inline typename Al::MPICUDABackend::req_type
get_request<Al::MPICUDABackend>() {
  return Al::MPICUDABackend::null_req;
}
