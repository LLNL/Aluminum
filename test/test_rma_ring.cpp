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

#include <iostream>
#include "Al.hpp"
#include "test_utils.hpp"

#include <stdlib.h>
#include <math.h>
#include <string>

size_t max_size = 1<<30;

template <typename Backend, typename T>
void test_rma_ring() {
  typename Backend::comm_type comm;  // Use COMM_WORLD.

  int rank = comm.rank();
  int np = comm.size();
  int rhs = (rank + 1) % np;
  int lhs = (rank - 1 + np) % np;

  // Compute sizes to test.
  std::vector<size_t> sizes = {0};
  for (size_t size = 1; size <= max_size; size *= 2) {
    sizes.push_back(size);
    // Avoid duplicating 2.
    if (size > 1) {
      sizes.push_back(size + 1);
    }
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    auto &&buf = get_vector<T, Backend>(size);
    auto &&ref = VectorType<T, Backend>::gen_data(size);
    float *rhs_buf = nullptr;
    float *lhs_buf = nullptr;
    if (rank % 2) {
      rhs_buf = Al::ext::AttachRemoteBuffer<Backend>(buf.data(), rhs, comm);
      lhs_buf = Al::ext::AttachRemoteBuffer<Backend>(buf.data(), lhs, comm);
    } else {
      lhs_buf = Al::ext::AttachRemoteBuffer<Backend>(buf.data(), lhs, comm);
      rhs_buf = Al::ext::AttachRemoteBuffer<Backend>(buf.data(), rhs, comm);
    }
    // Transfer from rank 0 to np-1
    if (rank == 0) {
      buf = ref;
    } else {
      Al::ext::Wait<Backend>(lhs, comm);
    }
    Al::ext::Put<Backend>(buf.data(), rhs, rhs_buf, size, comm);
    Al::ext::Notify<Backend>(rhs, comm);
    if (rank == 0) {
      Al::ext::Wait<Backend>(lhs, comm);
      if (!check_vector(VectorType<T, Backend>::copy_to_host(ref), buf)) {
        std::cout << "Ring transfer from LHS to RHS does not match" <<
            std::endl;
        std::abort();
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Transfer from rank np-1 to 0
    if (rank == np - 1) {
      buf = ref;
    } else {
      Al::ext::Wait<Backend>(rhs, comm);
    }
    Al::ext::Put<Backend>(buf.data(), lhs, lhs_buf, size, comm);
    Al::ext::Notify<Backend>(lhs, comm);
    if (rank == np - 1) {
      Al::ext::Wait<Backend>(rhs, comm);
      if (!check_vector(VectorType<T, Backend>::copy_to_host(ref), buf)) {
        std::cout << "Ring transfer from RHS to LHS does not match" <<
            std::endl;
        std::abort();
      }
    }
    Al::ext::DetachRemoteBuffer<Backend>(rhs_buf, rhs, comm);
    Al::ext::DetachRemoteBuffer<Backend>(lhs_buf, lhs, comm);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI-CUDA";
  if (argc >= 2) {
    backend = argv[1];
  }
  if (argc == 3) {
    max_size = std::stoul(argv[2]);
  }

  if (backend == "MPI") {
    std::cerr << "MPI backend is not supported" << std::endl;
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    test_rma_ring<Al::MPICUDABackend, float>();
#endif
  } else {
    std::cerr << "usage: " << argv[0] << " [";
#ifdef AL_HAS_MPI_CUDA
    std::cerr << "MPI-CUDA";
#endif
    std::cerr << "]" << std::endl;
    return -1;
  }

  Al::Finalize();
  return 0;
}
