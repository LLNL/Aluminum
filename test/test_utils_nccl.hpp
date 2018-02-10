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
std::vector<typename allreduces::NCCLBackend::algo_type>
get_nb_allreduce_algorithms<allreduces::NCCLBackend>() {
  // NCCLBackend does not have non-blocking interface implemented
  std::vector<typename allreduces::NCCLBackend::algo_type> algos = {};
  return algos;
}

