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

/**
 * This provides a simple "Hello, world" program in Aluminum, similar
 * to many MPI "Hello, world" examples. When run, each rank should
 * print out "Hello, world, from rank <rank> of <number of processes>."
 * This demonstrates simple initialization, communicator creation and
 * access, and finalization.
 *
 * Comments along the way explain details.
 */

// For printing.
#include <iostream>

// This is the main Aluminum header file that you should include to
// use the library.
#include <Al.hpp>
// Include some simple helper utilities.
#include "utils.hpp"

// Note: If Aluminum has been compiled with CUDA or HIP/ROCm support,
// the appropriate headers will be transitively included by Al.hpp
// (although there is no harm if you include them yourself).
// Aluminum provides the AL_HAS_CUDA macro to check whether there is
// CUDA support in Aluminum. If you build with ROCm support, this macro
// will *also* be set (because Aluminum mostly uses hipify for ROCm
// support). Should you need to check specifically for ROCm, there is a
// AL_HAS_ROCM macro.


// Select which Aluminum backend to use.
// Supported backends: MPIBackend, NCCLBackend, HostTransferBackend.
// Note: All Aluminum operations are templated on the backend.
using AlBackend = Al::MPIBackend;


int main(int argc, char** argv) {
  // If using CUDA/HIP, you should set the CUDA/HIP device before
  // initializing Aluminum. Aluminum expects there to be only one
  // process per GPU.
#if defined AL_HAS_CUDA || defined AL_HAS_ROCM
  // This simply uses cudaGetDeviceCount to determine the number of
  // GPUs. For testing, you can override that with the AL_NUM_GPUS
  // environment variable.
  const int num_gpus = get_number_of_gpus();
  // These attempt to determine the number of processes ont his node
  // and the calling process's rank on that node using environment
  // variables provided by common MPI runtimes/launchers.
  const int local_rank = get_local_rank();
  const int local_size = get_local_size();
  // Note: Only local_rank is needed for setting the device. num_gpus
  // and local_size are used for error checking.
  // Enforce that there cannot be more processes than GPUs.
  if (num_gpus < local_size) {
    std::cerr << "Number of available GPUs (" << num_gpus << ")"
              << " is smaller than the number of local MPI ranks "
              << "(" << local_size << ")" << std::endl;
    std::abort();
  }

  // Set the CUDA device.
  // AL_FORCE_CHECK_GPU_NOSYNC checks for CUDA errors and throws an
  // Aluminum exception if one occurs.
  // The "FORCE" means that the return value is always checked,
  // regardless of debug level.
  // The "NOSYNC" means that the check does not synchronize the
  // CUDA/HIP device beforehand, so it might also see errors from
  // earlier CUDA/HIP calls.
  // For general use, AL_CHECK_GPU is probably the right choice.
  AL_FORCE_CHECK_GPU_NOSYNC(AlGpuSetDevice(local_rank));
#endif  /** AL_HAS_CUDA || AL_HAS_ROCM */

  // Initialize Aluminum. Much like MPI, Aluminum takes argc and argv
  // as input. (Unlike MPI, it takes them by reference.)
  // Aluminum does not specify thread safety at runtime.
  // You must call this before making any other calls to Aluminum
  // (except Al::Initialized).
  Al::Initialize(argc, argv);

  // Create a new communicator for the backend.
  // With no arguments, the communicator will use MPI_COMM_WORLD.
  // If the backend is stream-aware (e.g., NCCL), it will also use the
  // default stream.
  // You can optionally specify an MPI communicator and a stream to
  // associate with the communicator. The MPI communicator will be
  // duplicated internally.
  typename AlBackend::comm_type comm;

  // Print our Hello, world message.
  std::cout << "Hello, world, from rank "
            << comm.rank() << " of "
            << comm.size() << "." << std::endl;

  // Note: Aluminum communicators are objects, not references.
  // Their destructors will clean them up when the last reference is
  // gone. Hence, you should pass them by reference, not value.
  // (Indeed, their copy constructors and assignment operators are
  // deleted to prevent unintentional copying. Use the copy() method
  // if you actually want a copy.)

  // This cleans up and finalizes Aluminum. No calls to Aluminum can
  // be made after calling this. (Except Al::Initialized.)
  Al::Finalize();

  return 0;
}
