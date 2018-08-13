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
  // Cache this (unlikely we run on devices with different frequencies.)
  static long long int freq_hz = 0;
  if (freq_hz == 0) {
    int device;
    cudaGetDevice(&device);
    int freq_khz;
    cudaDeviceGetAttribute(&freq_khz, cudaDevAttrClockRate, device);
    freq_hz = (long long int) freq_khz * 1000;  // Convert from KHz.
  }
  double cycles = length * freq_hz;
  wait_kernel<<<1, 1, 0, stream>>>((long long int) cycles);
}
