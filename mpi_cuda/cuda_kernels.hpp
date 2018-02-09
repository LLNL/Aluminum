#pragma once

namespace allreduces {
namespace internal {
namespace mpi_cuda {

void reduce1(float *dst, const float *src, size_t count,
             cudaStream_t s);
void reduce2(float *dst, const float *src, size_t count,
             cudaStream_t s);
void reduce4(float *dst, const float *src, size_t count,
             cudaStream_t s);
void reduce_thrust(float *dst, const float *src, size_t count,
                   cudaStream_t s);

} // namespace mpi_cuda
} // namespace internal
} // namespace allreduces

