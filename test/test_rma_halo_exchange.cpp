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
#include <sstream>

size_t min_size = 1;
size_t max_size = 1<<25;

void get_proc_idx(int rank, int px, int &rx, int &ry) {
  rx = rank % px;
  ry = rank / px;
  return;
}

int get_rank(int rx, int ry, int px) {
  if (rx == MPI_PROC_NULL || ry == MPI_PROC_NULL) {
    return MPI_PROC_NULL;
  }
  return rx + ry * px;
}

int get_rhs(int rank_dim, int dim_size) {
  int rhs = rank_dim + 1;
  if (rhs >= dim_size) {
    return MPI_PROC_NULL;
  }
  return rhs;
}

int get_lhs(int rank_dim, int) {
  int lhs = rank_dim - 1;
  if (lhs < 0) {
    lhs = MPI_PROC_NULL;
  }
  return lhs;
}

template <typename T>
T &get(std::vector<T> &vec, int dim, int side) {
  return vec[dim*2+side];
}

template <typename T>
T &get(std::vector<T> &vec, int dim) {
  return vec[dim];
}

template <typename Backend, typename T>
void test_rma_halo_exchange(int px, int py) {
  // Ugly hack.
  std::vector<typename Backend::comm_type> comms;
  comms.emplace_back(std::move(*(CommWrapper<Backend>(MPI_COMM_WORLD).comm_.release())));
  comms.emplace_back(std::move(*(CommWrapper<Backend>(MPI_COMM_WORLD).comm_.release())));
  comms.emplace_back(std::move(*(CommWrapper<Backend>(MPI_COMM_WORLD).comm_.release())));
  comms.emplace_back(std::move(*(CommWrapper<Backend>(MPI_COMM_WORLD).comm_.release())));
  typename Backend::comm_type &comm = comms[0];

  int rank = comm.rank();
  int np = comm.size();
  if (px * py != np) {
    std::cout << "Size of the 2D process grid does not match the size of number of MPI ranks." << std::endl;
    std::abort();
  }

  int pidx[2];
  get_proc_idx(rank, px, pidx[0], pidx[1]);

  std::vector<int> peer_ranks;
  peer_ranks.push_back(get_rank(get_rhs(pidx[0], px), pidx[1], px));
  peer_ranks.push_back(get_rank(get_lhs(pidx[0], px), pidx[1], px));
  peer_ranks.push_back(get_rank(pidx[0], get_rhs(pidx[1], py), px));
  peer_ranks.push_back(get_rank(pidx[0], get_lhs(pidx[1], py), px));

  // matching the communicators with RHS and LHS
  for (int dim = 0; dim < 2; ++dim) {
    if (pidx[dim] % 2) {
      std::cout << "Swapping comm for " << dim << " of rank " << rank << std::endl;
      std::swap(get(comms, dim, 0), get(comms, dim, 1));
    }
  }

  // Compute sizes to test.
  std::vector<size_t> sizes = {0};
  for (size_t size = min_size; size <= max_size; size *= 2) {
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
    std::vector<typename VectorType<T, Backend>::type> data({
        VectorType<T, Backend>::gen_data(size),
        VectorType<T, Backend>::gen_data(size),
        VectorType<T, Backend>::gen_data(size),
        VectorType<T, Backend>::gen_data(size)});
    auto ref(data);
    std::vector<typename VectorType<T, Backend>::type> tmp({
        get_vector<T, Backend>(size), get_vector<T, Backend>(size),
        get_vector<T, Backend>(size), get_vector<T, Backend>(size)});
    std::vector<float*> dest_data(4, nullptr);
    std::vector<float*> dest_tmp(4, nullptr);
    for (int dim = 0; dim < 2; ++dim) {
      if (pidx[dim] % 2) {
        for (int side = 0; side < 2; ++side) {
          get(dest_data, dim, side) = Al::ext::AttachRemoteBuffer<Backend>(
              get(data, dim, side).data(), get(peer_ranks, dim, side),
              get(comms, dim, side));
          get(dest_tmp, dim, side) = Al::ext::AttachRemoteBuffer<Backend>(
              get(tmp, dim, side).data(), get(peer_ranks, dim, side),
              get(comms, dim, side));
        }
      } else {
        for (int side = 1; side >= 0; --side) {
          get(dest_data, dim, side) = Al::ext::AttachRemoteBuffer<Backend>(
              get(data, dim, side).data(), get(peer_ranks, dim, side),
              get(comms, dim, side));
          get(dest_tmp, dim, side) = Al::ext::AttachRemoteBuffer<Backend>(
              get(tmp, dim, side).data(), get(peer_ranks, dim, side),
              get(comms, dim, side));
        }
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      if (pidx[dim] % 2) {
        for (int side = 0; side < 2; ++side) {
          Al::ext::Sync<Backend>(
              get(peer_ranks, dim, side), get(comms, dim, side));
        }
      } else {
        for (int side = 1; side >= 0; --side) {
          Al::ext::Sync<Backend>(
              get(peer_ranks, dim, side), get(comms, dim, side));
        }
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      for (int side = 0; side < 2; ++side) {
        Al::ext::Put<Backend>(
            get(data, dim, side).data(), get(peer_ranks, dim, side),
            get(dest_tmp, dim, side), size, get(comms, dim, side));
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      if (pidx[dim] % 2) {
        for (int side = 0; side < 2; ++side) {
          Al::ext::Sync<Backend>(
              get(peer_ranks, dim, side), get(comms, dim, side));
        }
      } else {
        for (int side = 1; side >= 0; --side) {
          Al::ext::Sync<Backend>(
              get(peer_ranks, dim, side), get(comms, dim, side));
        }
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      for (int side = 0; side < 2; ++side) {
        Al::ext::Put<Backend>(
            get(tmp, dim, side).data(), get(peer_ranks, dim, side),
            get(dest_data, dim, side), size, get(comms, dim, side));
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      if (pidx[dim] % 2) {
        for (int side = 0; side < 2; ++side) {
          Al::ext::Sync<Backend>(
              get(peer_ranks, dim, side), get(comms, dim, side));
        }
      } else {
        for (int side = 1; side >= 0; --side) {
          Al::ext::Sync<Backend>(
              get(peer_ranks, dim, side), get(comms, dim, side));
        }
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      for (int side = 0; side < 2; ++side) {
        if (!check_vector(VectorType<T, Backend>::copy_to_host(get(data, dim, side)), get(ref, dim, side))) {
          std::cout << "Buffer does not match on dimension " << dim << std::endl;
          std::abort();
        }
      }
    }
    for (int dim = 0; dim < 2; ++dim) {
      if (pidx[dim] % 2) {
        for (int side = 0; side < 2; ++side) {
          Al::ext::DetachRemoteBuffer<Backend>(
              get(dest_tmp, dim, side), get(peer_ranks, dim, side),
              get(comms, dim, side));
          Al::ext::DetachRemoteBuffer<Backend>(
              get(dest_data, dim, side), get(peer_ranks, dim, side),
              get(comms, dim, side));
        }
      } else {
        for (int side = 1; side >= 0; --side) {
          Al::ext::DetachRemoteBuffer<Backend>(
              get(dest_tmp, dim, side), get(peer_ranks, dim, side),
              get(comms, dim, side));
          Al::ext::DetachRemoteBuffer<Backend>(
              get(dest_data, dim, side), get(peer_ranks, dim, side),
              get(comms, dim, side));
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);
  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  std::string backend = "MPI-CUDA";
  int px = 1;
  int py = np;
  if (argc >= 2) {
    backend = argv[1];
  }
  if (argc >= 4) {
    px = atoi(argv[2]);
    py = atoi(argv[3]);
  }
  if (argc == 5) {
    min_size = std::stoul(argv[4]);
    max_size = std::stoul(argv[4]);
  }
  if (argc >= 6) {
    min_size = std::stoul(argv[4]);
    max_size = std::stoul(argv[5]);
  }

  if (backend == "MPI") {
    std::cerr << "MPI backend is not supported" << std::endl;
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    test_rma_halo_exchange<Al::MPICUDABackend, float>(px, py);
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
