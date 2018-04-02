#pragma once

#include "base.hpp"

namespace Al {
namespace internal {
namespace mpi_cuda {

enum class ReductionOperandType {
  UNKNOWN, INT, LONG, FLOAT, DOUBLE};

template <typename T>
struct GetReductionOperandType {
  static constexpr ReductionOperandType key =
      ReductionOperandType::UNKNOWN;
};
template <>
struct GetReductionOperandType<int> {
  static constexpr ReductionOperandType key =
      ReductionOperandType::INT;
};
template <>
struct GetReductionOperandType<long> {
  static constexpr ReductionOperandType key =
      ReductionOperandType::LONG;
};
template <>
struct GetReductionOperandType<float> {
  static constexpr ReductionOperandType key =
      ReductionOperandType::FLOAT;
};
template <>
struct GetReductionOperandType<double> {
  static constexpr ReductionOperandType key =
      ReductionOperandType::DOUBLE;
};

void reduce1(void *dst, const void *src, size_t count,
             cudaStream_t s, ReductionOperator op,
             ReductionOperandType type);
void reduce2(void *dst, const void *src, size_t count,
             cudaStream_t s, ReductionOperator op,
             ReductionOperandType type);
void reduce4(void *dst, const void *src, size_t count,
             cudaStream_t s, ReductionOperator op,
             ReductionOperandType type);
#if 0
void reduce_thrust(void *dst, const void *src, size_t count,
                   cudaStream_t s, ReductionOperator op);
#endif

} // namespace mpi_cuda
} // namespace internal
} // namespace Al
