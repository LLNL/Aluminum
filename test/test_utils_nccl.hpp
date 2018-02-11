#pragma once

#include "test_utils_cuda.hpp"

template <>
struct VectorType<allreduces::NCCLBackend> {
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
inline typename allreduces::NCCLBackend::req_type
get_request<allreduces::NCCLBackend>() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  return s;
}
