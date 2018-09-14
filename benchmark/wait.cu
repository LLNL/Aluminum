////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

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
