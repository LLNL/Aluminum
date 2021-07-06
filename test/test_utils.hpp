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

#pragma once

#include "Al.hpp"

#include <iostream>
#include <vector>
#include <limits>
#include <random>
#include <cstdlib>
#include <cxxopts.hpp>


/** Helper for generating random data. */
template <typename T, typename Generator,
          std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
T gen_random_val(Generator& g) {
  std::uniform_real_distribution<T> rng;
  return rng(g);
}
template <typename T, typename Generator,
          std::enable_if_t<std::is_integral<T>::value, bool> = true>
T gen_random_val(Generator& g) {
  std::uniform_int_distribution<T> rng;
  return rng(g);
}

/**
 * Identify the type of vector to be used for each backend.
 *
 * By default this is std::vector<T>.
 */
template <typename T, typename Backend>
struct VectorType {
  using type = std::vector<T>;

  /** Generate a vector of random data of size count. */
  static type gen_data(size_t count, int = 0) {
    static bool rng_seeded = false;
    static std::minstd_rand rng_gen;
    if (!rng_seeded) {
      // Seed using the MPI rank (only if MPI has been initialized).
      int flag;
      MPI_Initialized(&flag);
      if (flag) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        rng_gen.seed(rank + 1);
        rng_seeded = true;
      }
    }
    type v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = gen_random_val<T>(rng_gen);
    }
    return v;
  }

  /** Return a copy of the data on the host. */
  static std::vector<T> copy_to_host(const type& v) {
    return std::vector<T>(v);
  }
};

/** Return a vector of size count. */
template <typename T, typename Backend>
typename VectorType<T, Backend>::type get_vector(size_t count) {
  return typename VectorType<T, Backend>::type(count);
}

/**
 * Manager for streams for a backend.
 *
 * Default implementation for backends that do not have real streams.
 */
template <typename Backend>
struct StreamManager {
  /** Type of the underlying stream. */
  using StreamType = int;

  /** Initialize the stream manager to support this many streams. */
  static void init(size_t /*num_streams*/) {}
  /** Clean up the stream manager. */
  static void finalize() {}
  /** Return a new stream. */
  static StreamType get_stream() { return 0; };
};

/** RAII manager for a communicator. */
template <typename Backend>
struct CommWrapper {
  std::unique_ptr<typename Backend::comm_type> comm_;
  CommWrapper(MPI_Comm mpi_comm) :
    comm_(std::make_unique<typename Backend::comm_type>(mpi_comm)) {}
  CommWrapper(CommWrapper<Backend>&& other) = default;
  CommWrapper<Backend>& operator=(CommWrapper<Backend>&& other) = default;
  ~CommWrapper() {};
  typename Backend::comm_type& comm() { return *comm_; }
  const typename Backend::comm_type& comm() const { return *comm_; }
  int rank() const { return comm_->rank(); }
  int size() const { return comm_->size(); }
};

/**
 * Ensure all Aluminum operations on a communicator have completed.
 *
 * For backends that work with compute streams on other devices
 * (e.g., GPUs), this ensures the operations finish.
 */
template <typename Backend>
void complete_operations(typename Backend::comm_type&) {}

/**
 * Helper to hang for debugging.
 *
 * If non-negative, hangs that rank; if negative, hangs all ranks.
 */
void hang_for_debugging(int hang_rank) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (hang_rank < 0 || rank == hang_rank) {
    if (hang_rank < 0 && rank == 0) {
      std::cout << "Hanging all ranks" << std::endl;
    } else if (rank == hang_rank) {
      std::cout << "Hanging rank " << rank << std::endl;
    }
    volatile bool hang = true;
    while (hang) {}
  }
}

/** Helper for dispatch_from_args, handles type dispatch. */
template <typename Backend, typename F>
void dispatch_to_backend_type_helper(cxxopts::ParseResult& parsed_opts,
                                     F functor) {
  const std::unordered_map<std::string, std::function<void()>> dispatch = {
    {"char", [&]() { functor.template operator()<Backend, char>(parsed_opts); } },
    {"schar", [&]() { functor.template operator()<Backend, signed char>(parsed_opts); } },
    {"uchar", [&]() { functor.template operator()<Backend, unsigned char>(parsed_opts); } },
    {"short", [&]() { functor.template operator()<Backend, short>(parsed_opts); } },
    {"ushort", [&]() { functor.template operator()<Backend, unsigned short>(parsed_opts); } },
    {"int", [&]() { functor.template operator()<Backend, int>(parsed_opts); } },
    {"uint", [&]() { functor.template operator()<Backend, unsigned int>(parsed_opts); } },
    {"long", [&]() { functor.template operator()<Backend, long>(parsed_opts); } },
    {"ulong", [&]() { functor.template operator()<Backend, unsigned long>(parsed_opts); } },
    {"longlong", [&]() { functor.template operator()<Backend, long long>(parsed_opts); } },
    {"ulonglong", [&]() { functor.template operator()<Backend, unsigned long long>(parsed_opts); } },
    {"float", [&]() { functor.template operator()<Backend, float>(parsed_opts); } },
    {"double", [&]() { functor.template operator()<Backend, double>(parsed_opts); } },
    {"longdouble", [&]() { functor.template operator()<Backend, long double>(parsed_opts); } },
#ifdef AL_HAS_NCCL
    {"half", [&]() { functor.template operator()<Backend, __half>(parsed_opts); } },
#endif
  };
  auto datatype = parsed_opts["datatype"].as<std::string>();
  auto i = dispatch.find(datatype);
  if (i == dispatch.end()) {
    std::cerr << "Unknown datatype " << datatype << std::endl;
    std::abort();
  }
  i->second();
}

/**
 * Run a functor with a backend and type given in parsed_opts.
 *
 * Excepts a "backend" and "datatype" argument.
 */
template <typename F>
void dispatch_to_backend(cxxopts::ParseResult& parsed_opts, F functor) {
  const std::unordered_map <std::string, std::function<void()>> dispatch = {
    {"mpi", [&](){ dispatch_to_backend_type_helper<Al::MPIBackend>(parsed_opts, functor); } },
#ifdef AL_HAS_NCCL
    {"nccl", [&](){ dispatch_to_backend_type_helper<Al::NCCLBackend>(parsed_opts, functor); } },
#endif
#ifdef AL_HAS_HOST_TRANSFER
    {"ht", [&](){ dispatch_to_backend_type_helper<Al::HostTransferBackend>(parsed_opts, functor); } },
#endif
  };
  auto backend = parsed_opts["backend"].as<std::string>();
  auto i = dispatch.find(backend);
  if (i == dispatch.end()) {
    std::cerr << "Unsupported backend " << backend << std::endl;
    std::abort();
  }
  i->second();
}

#ifdef AL_HAS_CUDA

/**
 * Return the number of GPUs to use on the system.
 *
 * By default this will use CUDA to determine how many GPUs there are.
 * This can be overridden using the AL_NUM_GPUS environment variable.
 */
inline int get_number_of_gpus() {
  int num_gpus = 0;
  char* env = std::getenv("AL_NUM_GPUS");
  if (env) {
    num_gpus = std::atoi(env);
    if (num_gpus == 0) {
      std::cerr << "AL_NUM_GPUS either 0 or invalid value: "
                << env << std::endl;
      std::abort();
    }
  } else {
    AL_FORCE_CHECK_CUDA_NOSYNC(cudaGetDeviceCount(&num_gpus));
  }
  return num_gpus;
}

#endif  /** AL_HAS_CUDA */

/** Attempt to identify the local rank on a node from the environment. */
inline int get_local_rank() {
  char* env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) {
    env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  }
  if (!env) {
    env = std::getenv("SLURM_LOCALID");
  }
  if (!env) {
    std::cerr << "Cannot determine local rank" << std::endl;
    std::abort();
  }
  return std::atoi(env);
}

/** Attempt to identify the number of ranks on a node from the environment. */
inline int get_local_size() {
  char *env = std::getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env) {
    env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  }
  if (!env) {
    env = std::getenv("SLURM_NTASKS_PER_NODE");
  }
  if (!env) {
    std::cerr << "Cannot determine local size" << std::endl;
    std::abort();
  }
  return std::atoi(env);
}

#ifdef AL_HAS_CUDA

/**
 * Attempt to automatically set the CUDA device to something reasonable.
 *
 * Returns the index of the device that was selected.
 */
inline int set_device() {
  const int num_gpus = get_number_of_gpus();
  const int local_rank = get_local_rank();
  const int local_size = get_local_size();
  if (num_gpus < local_size) {
    std::cerr << "Number of available GPUs (" << num_gpus << ")"
              << " is smaller than the number of local MPI ranks"
              << "(" << local_size << ")" << std::endl;
    std::abort();
  }
  AL_FORCE_CHECK_CUDA_NOSYNC(cudaSetDevice(local_rank));
  return local_rank;
}

#endif  /** AL_HAS_CUDA */

/** Attempt to nicely initialize Aluminum. */
inline void test_init_aluminum(int& argc, char**& argv) {
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);
}

/** Attempt to nicely finalize Aluminum. */
inline void test_fini_aluminum() {
  Al::Finalize();
}

/** Compare two vectors with a given tolerance. */
template <typename T>
bool check_vector(const std::vector<T> &expected,
                  const std::vector<T> &actual,
                  size_t start = 0,
                  size_t end = std::numeric_limits<size_t>::max(),
                  const T eps = T(1e-4)) {
  if (end == std::numeric_limits<size_t>::max()) {
    end = expected.size();
  }
  if (expected.size() != actual.size()) {
    return false;
  }
  for (size_t i = start; i < end; ++i) {
    const T diff = (expected[i] > actual[i])
      ? expected[i] - actual[i]
      : actual[i] - expected[i];
    if (diff > eps) {
      return false;
    }
  }
  return true;
}

/**
 * Return a set of sizes to test between start_size and max_size (inclusive).
 *
 * If odds is true, ensure odd-numbered sizes are generated.
 */
std::vector<size_t> get_sizes(size_t start_size, size_t max_size,
                              bool odds = false) {
  std::vector<size_t> sizes;
  if (start_size == 0) {
    sizes.push_back(0);
    start_size = 1;
  }
  for (size_t size = start_size; size <= max_size; size *= 2) {
    sizes.push_back(size);
    if (odds && size > 1) {
      sizes.push_back(size + 1);
    }
  }
  return sizes;
}

/**
 * Helper to call get_sizes based on parsed options.
 *
 * Expects the options "size", "min-size", and "max-size".
 */
std::vector<size_t> get_sizes_from_opts(cxxopts::ParseResult& parsed_opts) {
  size_t min_size = parsed_opts["min-size"].as<size_t>();
  size_t max_size = parsed_opts["max-size"].as<size_t>();
  if (parsed_opts.count("size")) {
    min_size = parsed_opts["size"].as<size_t>();
    max_size = min_size;
  }
  return get_sizes(min_size, max_size);
}

/** Return a human-readable string for size. */
std::string human_readable_size(size_t size_) {
  double size = static_cast<double>(size_);
  if (size < 1024) {
    return std::to_string(size);
  }
  size /= 1024;
  if (size < 1024) {
    return std::to_string(size) + " K";
  }
  size /= 1024;
  if (size < 1024) {
    return std::to_string(size) + " M";
  }
  size /= 1024;
  return std::to_string(size) + " G";
}

// Pull in relevant headers for simplicity.
#include "op_dispatcher.hpp"
#include "test_utils_mpi.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_ht.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif
