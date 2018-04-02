#pragma once

#include <cstdlib>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "mpi_cuda/coll_util.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

void gpu_reduce_to_host(const std::vector<int> &gpus,
                        float *host_buf,
                        const std::vector<float *> &gbufs,
                        size_t count,
                        std::vector<cudaStream_t> &streams);

void gpu_reduce_to_host_pipeline(const std::vector<int> &gpus,
                                 float *host_buf,
                                 const std::vector<float *> &gpu_bufs,
                                 size_t count,
                                 std::vector<cudaStream_t> &streams1,
                                 std::vector<cudaStream_t> &streams2);

void gpu_allreduce(const std::vector<int> &gpus,
                   const std::vector<float *> &gbufs,
                   size_t count,
                   std::vector<cudaStream_t> &streams);

void gpu_allreduce2(const std::vector<int> &gpus,
                    const std::vector<float *> &gpu_data,
                    size_t count,
                    std::vector<cudaStream_t> &streams);


void gpu_broadcast(const std::vector<int> &gpus,
                   float *host_buf,                   
                   const std::vector<float *> &gbufs,
                   size_t count,
                   std::vector<cudaStream_t> &streams);

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
