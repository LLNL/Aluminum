#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

/** Cause the stream to wait for length seconds. */
void gpu_wait(double length, cudaStream_t stream);
