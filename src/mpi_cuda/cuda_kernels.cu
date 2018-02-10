#include <cuda_runtime.h>
//#include <thrust/transform.h>
//#include <thrust/device_ptr.h>
#include "mpi_cuda/cuda_kernels.hpp"

namespace allreduces {
namespace internal {
namespace mpi_cuda {

template <typename T, ReductionOperator op>
struct BinaryOp;

template <typename T>
struct BinaryOp<T, ReductionOperator::sum> {
  __device__ static T calc(const T& x, const T& y) {
    return x + y;
  }
};

template <typename T>
struct BinaryOp<T, ReductionOperator::prod> {
  __device__ static T calc(const T& x, const T& y) {
    return x * y;
  }
};

template <typename T>
struct BinaryOp<T, ReductionOperator::min> {
  __device__ static T calc(const T& x, const T& y) {
    return (x < y ? x : y);
  }
};

template <typename T>
struct BinaryOp<T, ReductionOperator::max> {
  __device__ static T calc(const T& x, const T& y) {
    return (x > y ? x : y);
  }
};

template <typename T, int len>
struct ShortVectorType {
  using type = T;
};

template <>
struct ShortVectorType<float, 1> {
  using type = float;
};
template <>
struct ShortVectorType<float, 2> {
  using type = float2;
};
template <>
struct ShortVectorType<float, 4> {
  using type = float4;
};

template <>
struct ShortVectorType<double, 1> {
  using type = double;
};
template <>
struct ShortVectorType<double, 2> {
  using type = double2;
};
template <>
struct ShortVectorType<double, 4> {
  using type = double4;
};

template <>
struct ShortVectorType<int, 1> {
  using type = int;
};
template <>
struct ShortVectorType<int, 2> {
  using type = int2;
};
template <>
struct ShortVectorType<int, 4> {
  using type = int4;
};

template <>
struct ShortVectorType<long, 1> {
  using type = long;
};
template <>
struct ShortVectorType<long, 2> {
  using type = long2;
};
template <>
struct ShortVectorType<long, 4> {
  using type = long4;
};

template <ReductionOperator op, typename T, int VectorLen>
struct ReduceKernel;

template <ReductionOperator op, typename T>
struct ReduceKernel<op, T, 1> {
  __device__ static void kernel(void *dst, const void *src, size_t count) {
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= count) return;
    T *dst_t = static_cast<T*>(dst);
    const T *src_t = static_cast<const T*>(src);
    dst_t[offset] = BinaryOp<T, op>::calc(dst_t[offset], src_t[offset]);
  }
};

template <ReductionOperator op, typename T>
struct ReduceKernel<op, T, 2> {
  __device__ static void kernel(void *dst, const void *src, size_t count) {
    using VectorT = typename ShortVectorType<T, 2>::type;
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= count) return;
    VectorT *dst_vector = static_cast<VectorT*>(dst);
    const VectorT *src_vector = static_cast<const VectorT*>(src);
    VectorT d = dst_vector[offset];
    VectorT s = src_vector[offset];
    d.x = BinaryOp<T, op>::calc(d.x, s.x);
    d.y = BinaryOp<T, op>::calc(d.y, s.y);  
    dst_vector[offset] = d;
  }
};

template <ReductionOperator op, typename T>
struct ReduceKernel<op, T, 4> {
  __device__ static void kernel(void *dst, const void *src,
                                size_t count) {
    using VectorT = typename ShortVectorType<T, 4>::type;    
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= count) return;
    VectorT *dst_vector = static_cast<VectorT*>(dst);
    const VectorT *src_vector = static_cast<const VectorT*>(src);
    VectorT d = dst_vector[offset];
    VectorT s = src_vector[offset];
    d.x = BinaryOp<T, op>::calc(d.x, s.x);
    d.y = BinaryOp<T, op>::calc(d.y, s.y);  
    d.z = BinaryOp<T, op>::calc(d.z, s.z);
    d.w = BinaryOp<T, op>::calc(d.w, s.w);  
    dst_vector[offset] = d;
  }
};

template <ReductionOperator op, typename T, int VectorLen>
__global__ void reduce_kernel(void *dst, const void *src, size_t count) {
  ReduceKernel<op, T, VectorLen>::kernel(dst, src, count);
}
  

template <ReductionOperator op, typename T, int VectorLen>
void reduce_v(void *dst, const void *src, size_t count, cudaStream_t s) {
  using VectorT = typename ShortVectorType<T, VectorLen>::type;
  if (count == 0) return;
  int tb_dim = 256;
  count /= VectorLen;
  int grid_dim = count / tb_dim + (count % tb_dim ? 1 : 0);
  reduce_kernel<op, T, VectorLen><<<grid_dim, tb_dim, 0, s>>>(
      dst, src, count);
}

// Passing each function argument to a template parameter

template <typename T, int VectorLen>
void reduce_v(void *dst, const void *src, size_t count,
              cudaStream_t s, ReductionOperator op) {
  switch (op) {
    case ReductionOperator::sum:
      reduce_v<ReductionOperator::sum, T, VectorLen>(dst, src, count, s);
      break;
    case ReductionOperator::prod:
      reduce_v<ReductionOperator::prod, T, VectorLen>(dst, src, count, s);
      break;
    case ReductionOperator::min:
      reduce_v<ReductionOperator::min, T, VectorLen>(dst, src, count, s);
      break;
    case ReductionOperator::max:
      reduce_v<ReductionOperator::max, T, VectorLen>(dst, src, count, s);
      break;
    default:
      throw_allreduce_exception("Unknown reduction operator");
  }
}

template <int VectorLen>
void reduce_v(void *dst, const void *src, size_t count,
              cudaStream_t s, ReductionOperator op,
              ReductionOperandType type) {
  switch (type) {
    case ReductionOperandType::INT:
      reduce_v<int, VectorLen>(dst, src, count, s, op);
      break;
    case ReductionOperandType::LONG:
      reduce_v<long, VectorLen>(dst, src, count, s, op);
      break;
    case ReductionOperandType::FLOAT:
      reduce_v<float, VectorLen>(dst, src, count, s, op);
      break;
    case ReductionOperandType::DOUBLE:
      reduce_v<double, VectorLen>(dst, src, count, s, op);
      break;
    default:
      throw_allreduce_exception("Unknown operand type");
  }
}

void reduce1(void *dst, const void *src, size_t count,
             cudaStream_t s, ReductionOperator op,
             ReductionOperandType type) {
  reduce_v<1>(dst, src, count, s, op, type);
}

void reduce2(void *dst, const void *src, size_t count,
             cudaStream_t s, ReductionOperator op,
             ReductionOperandType type) {
  if ((count % 2) == 0) {
    reduce_v<2>(dst, src, count, s, op, type);
  } else {
    reduce1(dst, src, count, s, op, type);
  }
}

void reduce4(void *dst, const void *src, size_t count,
             cudaStream_t s, ReductionOperator op,
             ReductionOperandType type) {
  if ((count % 4) == 0) {
    reduce_v<4>(dst, src, count, s, op, type);
  } else {
    reduce1(dst, src, count, s, op, type);
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

