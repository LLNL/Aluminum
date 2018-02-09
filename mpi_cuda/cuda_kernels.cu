#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>

namespace allreduces {
namespace internal {
namespace mpi_cuda {

__global__ void reduce_kernel(float *dst, const float *src, size_t count) {
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  //printf("dst[%d]: %d, src[%d]: %d\n", (int)offset, (int)dst[offset],
  //(int)offset, (int)src[offset]);
  dst[offset] += src[offset];  
  //printf("dst[%d]: %d, src[%d]: %d\n", (int)offset, (int)dst[offset],
  //(int)offset, (int)src[offset]);
}

__global__ void reduce_kernel(float2 *dst, const float2 *src,
                              size_t count) {
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  float2 d = dst[offset];
  float2 s = src[offset];
  d.x += s.x;
  d.y += s.y;
  dst[offset] = d;
}

__global__ void reduce_kernel(float4 *dst, const float4 *src,
                              size_t count) {
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  float4 d = dst[offset];
  float4 s = src[offset];
  d.x += s.x;
  d.y += s.y;
  d.z += s.z;
  d.w += s.w;
  dst[offset] = d;
}

template <typename VT>
void reduce_v(float *dst, const float *src, size_t count,
              cudaStream_t s) {
  if (count == 0) return;
  int tb_dim = 256;
  int vs = sizeof(VT) / sizeof(float);
  count /= vs;
  int grid_dim = count / tb_dim + (count % tb_dim ? 1 : 0);
  //std::cout << "count: " << count << ", grid_dim " << grid_dim << "\n";
  reduce_kernel<<<grid_dim, tb_dim, 0, s>>>((VT*)dst,
                                            (const VT*)src,
                                            count);
}

void reduce1(float *dst, const float *src, size_t count,
             cudaStream_t s) {
  reduce_v<float>(dst, src, count , s);
}

void reduce2(float *dst, const float *src, size_t count,
             cudaStream_t s) {
  if ((count % 2) == 0) {
    reduce_v<float2>(dst, src, count, s);
  } else {
    reduce1(dst, src, count, s);
  }
}

void reduce4(float *dst, const float *src, size_t count,
             cudaStream_t s) {
  if ((count % 4) == 0) {
    reduce_v<float4>(dst, src, count, s);
  } else {
    reduce1(dst, src, count, s);
  }
}
#if 0
void reduce_thrust(float *dst, const float *src, size_t count,
                   cudaStream_t s) {
  thrust::device_ptr<float> dst_ptr(dst);
  thrust::device_ptr<const float> src_ptr(src);
  thrust::transform(thrust::cuda::par.on(s),
                    src_ptr, src_ptr + count,
                    dst_ptr, dst_ptr,
                    thrust::plus<float>());
}
#endif


} // namespace mpi_cuda
} // namespace internal
} // namespace allreduces

