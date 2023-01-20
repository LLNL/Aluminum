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
 * This provides a simple exaple of using an Aluminum allreduce.
 * A buffer of random data is generated then allreduced and the zeroth
 * rank will print the whole vector.
 */

#include <Al.hpp>
#include "utils.hpp"


// Select which Aluminum backend to use.
using AlBackend = Al::MPIBackend;
// Type of data to be allreduced.
using DataType = float;
// Number of elements of DataType to be allreduced.
constexpr size_t num_elements = 4;


// A general note:
// Aluminum performs communication in a "stream-aware" manner.
// That is, if a backend has a notion of separate compute streams,
// e.g., for GPUs, communication will be synchronized with respect to
// that backend, and *not* the calling host code.
//
// Hence, for a GPU-aware backend (NCCL, HostTransfer), Aluminum
// operations follow standard CUDA/ROCm semantics, and the host-side
// call will complete after the operation has been enqueued. It is up
// to you to ensure appropriate synchronization.
//
// The MPI backend does not strictly have any streams associated with
// it, but you can think of it as having a default stream which is the
// calling CPU.


int main(int argc, char** argv) {
  // Initialize Aluminum.
#if defined AL_HAS_CUDA || defined AL_HAS_ROCM
  const int num_gpus = get_number_of_gpus();
  const int local_rank = get_local_rank();
  const int local_size = get_local_size();
  if (num_gpus < local_size) {
    std::cerr << "Number of available GPUs (" << num_gpus << ")"
              << " is smaller than the number of local MPI ranks "
              << "(" << local_size << ")" << std::endl;
    std::abort();
  }
  AL_FORCE_CHECK_GPU_NOSYNC(AlGpuSetDevice(local_rank));
#endif  /** AL_HAS_CUDA || AL_HAS_ROCM */
  Al::Initialize(argc, argv);

  // Create our communicator.
  // Stream-aware backends will use the default stream.
  // If you want to create a communicator with a stream, the stream
  // must be associated with the communicator at creation.
  // You can do so using `comm(mpi_comm, stream)`.
  // You can also duplicate an existing communicator and associate it
  // with a different stream using `comm.copy(new_stream)`.
  typename AlBackend::comm_type comm;

  // Create a buffer of random data on the appropriate device for the
  // backend. For MPI, this will be on CPU; for NCCL and HostTransfer,
  // this will be CUDA memory on the current device.
  // If the backend does not use streams, the comm.get_stream() call
  // will essentially be a nop.
  typename VectorType<DataType, AlBackend>::type data_vec =
    VectorType<DataType, AlBackend>::gen_data(num_elements, comm.get_stream());

  // Get a pointer to the data buffer. (This may be a device pointer.)
  DataType* buffer = data_vec.data();

  // Perform the allreduce using a summation operator.
  // This allreduce is in-place, so the result will be placed in the
  // same buffer as the input (`buffer` here).
  Al::Allreduce<AlBackend>(
    buffer, num_elements, Al::ReductionOperator::sum, comm);

  // Aluminum operations on compute streams can run asynchronously from
  // the host. You therefore may need to ensure they complete before
  // accessing the data, or otherwise ensure synchronization.
  // (Technical note 1: Aluminum respects stream ordering semantics, so
  // in general you can just enqueue subsequent operations as normal.)
  // (Technical note 2: Consequently, this synchronization is not
  // necessary here, since the copy out will be enqueued after the
  // Aluminum operation, but it's here for illustrative purposes.)
  complete_operations<AlBackend>(comm);

  // Move the allreduced data to the host, if necessary.
  std::vector<DataType> host_data =
    VectorType<DataType, AlBackend>::copy_to_host(data_vec);

  // Have the zeroth rank print the vector.
  // Other ranks will wait in the barrier until completion.
  // Note: We directly use MPI for the barrier because an Aluminum
  // barrier would synchronize computation with respect to the compute
  // stream, which may not be the host.
  // (Note: Should you need it, the underlying MPI communicator
  // associated with an Aluminum communicator can be accessed using
  // `comm.get_comm()`.)
  if (comm.rank() == 0) {
    std::cout << "Allreduced data: ";
    for (const auto& v : host_data) {
      std::cout << v << " ";
    }
    std::cout << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Note: The allocated data will be cleaned up automatically.

  // Clean up Aluminum.
  Al::Finalize();

  return 0;
}
