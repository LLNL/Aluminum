#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void wait_kernel(long long int cycles) {
  // Doesn't handle the clock wrapping.
  // Seems to wait longer than expected, but not an issue right now.
  const long long int start = clock64();
  long long int cur;
  do {
    cur = clock64();
  } while (cur - start < cycles);
}

}  // anonymous namespace

void gpu_wait(double length, cudaStream_t stream) {
  // Need to figure out frequency to convert seconds to cycles.
  // Might not be exactly accurate (especially w/ dynamic frequencies).
  int device;
  cudaGetDevice(&device);
  int freq_khz;  // Frequency is in KHz.
  cudaDeviceGetAttribute(&freq_khz, cudaDevAttrClockRate, device);
  double cycles = length * freq_khz*1000.0;
  wait_kernel<<<1, 1, 0, stream>>>((long long int) cycles);
}
