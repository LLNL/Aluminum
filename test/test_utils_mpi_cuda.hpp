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
inline void start_timer<Al::MPICUDABackend>(typename Al::MPICUDABackend::comm_type& comm) {
  cudaEvent_t start = get_timer_events().first;
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventRecord(start, comm.get_stream()));
}

template <>
inline double finish_timer<Al::MPICUDABackend>(typename Al::MPICUDABackend::comm_type& comm) {
  std::pair<cudaEvent_t, cudaEvent_t> events = get_timer_events();
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventRecord(events.second, comm.get_stream()));
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventSynchronize(events.second));
  float elapsed;
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaEventElapsedTime(&elapsed, events.first, events.second));
  return elapsed/1000.0;  // ms -> s
}

template <>
std::vector<typename Al::MPICUDABackend::algo_type>
get_allreduce_algorithms<Al::MPICUDABackend>() {
  std::vector<typename Al::MPICUDABackend::algo_type> algos = {
    Al::MPICUDABackend::algo_type::ring,    
    Al::MPICUDABackend::algo_type::bi_ring,
    Al::MPICUDABackend::algo_type::host_transfer
  };
  return algos;
}

template <>
std::vector<typename Al::MPICUDABackend::algo_type>
get_nb_allreduce_algorithms<Al::MPICUDABackend>() {
  std::vector<typename Al::MPICUDABackend::algo_type> algos = {
    Al::MPICUDABackend::algo_type::ring,
    Al::MPICUDABackend::algo_type::bi_ring,
    Al::MPICUDABackend::algo_type::host_transfer,
  };
  return algos;
}

template <>
inline typename Al::MPICUDABackend::req_type
get_request<Al::MPICUDABackend>() {
  return Al::MPICUDABackend::null_req;
}
