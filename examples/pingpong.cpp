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
 * This provides a simple example of a ping-pong using Aluminum point-
 * to-point operations. Ranks pair up and send data back and forth.
 */

#include <Al.hpp>
#include "utils.hpp"


// Select which Aluminum backend to use.
using AlBackend = Al::MPIBackend;
// Type of data to be sent.
using DataType = float;
// Number of elements of DataType to be sent.
constexpr size_t num_elements = 4;
// Number of iterations of pingpong.
constexpr size_t num_iters = 4;


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
#endif /** AL_HAS_CUDA */
  Al::Initialize(argc, argv);

  // Create our communicator.
  typename AlBackend::comm_type comm;

  // For this example, we require there be an even number of ranks so
  // each rank has a partner.
  if (comm.size() % 2 != 0) {
    std::cerr << "Pingpong requires an even number of ranks, but there are "
              << comm.size()
              << " ranks present."
              << std::endl;
    // Exit here.
    Al::Finalize();
    return 1;
  }

  // Create our buffer to pingpong.
  // Even ranks will start with the data, odd ranks with an unitialized
  // buffer.
  typename VectorType<DataType, AlBackend>::type data_vec =
    (comm.rank() % 2 == 0)
    ? VectorType<DataType, AlBackend>::gen_data(num_elements, comm.get_stream())
    : get_vector<DataType, AlBackend>(num_elements);

  // Get a pointer to the data buffer. (This may be a device pointer.)
  DataType* buffer = data_vec.data();

  // Do our pingpong iterations.
  // Flag for whether this rank is currently sending data.
  bool sending = comm.rank() % 2 == 0;
  // Compute this rank's partner.
  int partner = (comm.rank() % 2 == 0) ? comm.rank() + 1 : comm.rank() - 1;
  for (size_t iter = 0; iter < num_iters; ++iter) {
    if (sending) {
      // The rank sends its buffer to partner.
      Al::Send<AlBackend>(buffer, num_elements, partner, comm);
    } else {
      // The rank receives a buffer from partner.
      Al::Recv<AlBackend>(buffer, num_elements, partner, comm);
    }
    // Ensure the operation completes.
    complete_operations<AlBackend>(comm);
    // Have rank 0 print something.
    if (comm.rank() == 0) {
      if (iter % 2 == 0) {
        std::cout << "Ping!" << std::endl;
      } else {
        std::cout << "Pong!" << std::endl;
      }
    }
    sending = !sending;  // Swap roles.
  }

  // Clean up Aluminum.
  Al::Finalize();

  return 0;
}
