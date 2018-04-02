#pragma once

#include "test_utils_cuda.hpp"

template <>
struct VectorType<Al::NCCLBackend> {
  using type = CUDAVector<float>;
};


template <>
typename VectorType<Al::NCCLBackend>::type
gen_data<Al::NCCLBackend>(size_t count) {
  auto &&host_data = gen_data<Al::MPIBackend>(count);
  CUDAVector<float> data(host_data);
  return data;
}

template <>
inline typename Al::NCCLBackend::req_type
get_request<Al::NCCLBackend>() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  return s;
}

void get_expected_nccl_result_allreduce(CUDAVector<float>& input){
  std::vector<float> &&host_data = input.copyout();
  std::vector<float> recv(input.size());
  MPI_Allreduce(MPI_IN_PLACE, host_data.data(), input.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  input.copyin(host_data);
}

void get_expected_nccl_result_reduce(CUDAVector<float>& input) {
  std::vector<float> &&host_data = input.copyout();
  std::vector<float> recv(input.size());
  MPI_Reduce(host_data.data(), recv.data(), input.size(),
                MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  input.copyin(recv);
}

void get_expected_nccl_result_allgather(CUDAVector<float>& data, CUDAVector<float>&  expected) {
  std::vector<float> &&host_data = data.copyout();
  std::vector<float> recv(expected.size());
  MPI_Allgather(host_data.data(), data.size(), MPI_FLOAT, recv.data(), data.size(), MPI_FLOAT, MPI_COMM_WORLD);
  expected.copyin(recv);
}


void get_expected_nccl_result_reduce_scatter(CUDAVector<float>& data, CUDAVector<float>&  expected) {
  std::vector<float> &&host_data = data.copyout(); //#procs*count
  std::vector<float> recv(expected.size()); //count

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int num_procs = host_data.size()/expected.size();
  if(num_procs*expected.size() != host_data.size()){
    std::cout << rank << ": fatal error in reduce_scatter test: data and expected buffer sizes do not match" <<
        std::endl;
    std::abort();
  }

  std::vector<int> recvcounts (num_procs);
  for(int i=0; i<num_procs; i++)
    recvcounts[i] = (int) expected.size();

  MPI_Reduce_scatter (host_data.data(), recv.data(), recvcounts.data(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  expected.copyin(recv);
}

void get_expected_nccl_result_bcast(CUDAVector<float>& input) {
  std::vector<float> &&host_data = input.copyout();
  MPI_Bcast(host_data.data(), input.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
  input.copyin(host_data);
}
