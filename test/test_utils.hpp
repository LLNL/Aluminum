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

// This is the path of least resistance for coping with a convoluted
// FP16/HIP/hipCUB/rocPRIM/RCCL/etc issue. It's easier for now to just
// do this than to push for the proper fixes.
//
// The primary issue is that these operators are declared __device__
// when compiling as a "HIP" language file (i.e., when "-x hip" is
// passed to the compiler). Therefore, the proper overload cannot be
// resolved by the compiler. We just add these simple definitions and
// we're good to go from a compiler perspective (I don't know what
// performance cost this incurs, but we have historically suggested
// NOT using FP16 on the host side (or using a different library to
// support that)).
//
// So, at this point, you might be thinking "but this is not device
// code? Why is it being compiled as HIP code?", and you'd be right to
// be thinking that. Indeed, it is not device code, and for its part,
// nothing in this code requires HIP device support. Except that it
// #includes hipCUB stuff. For better or worse, AMD has hard-coded
// "hip::device" into their usage requirements for rocPRIM (via
// roc::rocprim_hip), which is a usage requirement for hipCUB. So we
// actually compile all of Aluminum as HIP code.
//
// This unveils another interesting discrepancy. When CMake introduced
// first-class language support for HIP, a new set of IMPORTED targets
// was introduced: hip-lang::host and hip-lang::device. Naturally,
// these largely seem to mirror hip::host and hip::device (imported
// via find_package(hip CONFIG)) except that the hip-lang IMPORTED
// targets seem to take more care to use Generator Expressions to
// isolate the flags they add to the detected compilation language for
// the source file in question. If we were to use hip-lang::device
// instead of hip::device, the only real benefit is that the compile
// line looks slightly saner (e.g., "-x hip" is not on it, nor are any
// of the GPU-related flags); the resulting error messages look worse,
// however.
//
// As it happens, hipCUB/rocPRIM headers rely on these
// "hip::device"-related command-line options being present as they
// seem to use some compiler-specific extensions (e.g.,
// "__align__(b)", where "b" is an int) that are not enabled unless
// using "-x hip".
//
// Also related: It is required the "-D__HIP_PLATFORM_HCC__=1" be on
// the command line; "-D__HIP_PLATFORM_AMD__=1" is NOT sufficient for
// hipCUB. If that define is not on the command line, hipCUB will have
// essentially no symbols, and the compiler will not know about
// "hipcub::CachingDeviceAllocator".
//
// Rather than crawling through this rat's nest of CMake/compiler CLI
// garbage, it seems like a better use of our time to just add these
// and move on...
//
// I should also note that this is not a problem for CUDA CUB. Its
// memory allocator file is completely host-side and does not have the
// same usage requirements imposed upon it, whether by CMake or by
// C++/CUDA. Ideally, HIP will adopt the same stream-aware memory
// allocator interface introduced in CUDA 11, and we can move to that
// universally (since we don't actually use the CUB features of
// CUB...).

#if defined(AL_HAS_ROCM) && defined(AL_HAS_HALF)

inline std::ostream& operator<<(std::ostream& os, __half const& x) {
  return os << static_cast<float>(x);
}

inline bool operator<(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) < static_cast<float>(rhs);
}
inline bool operator<=(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) <= static_cast<float>(rhs);
}
inline bool operator>(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) > static_cast<float>(rhs);
}
inline bool operator>=(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) >= static_cast<float>(rhs);
}
inline bool operator==(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) == static_cast<float>(rhs);
}
inline bool operator!=(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) != static_cast<float>(rhs);
}
inline __half operator-(__half const& lhs, __half const& rhs) {
  return static_cast<float>(lhs) - static_cast<float>(rhs);
}

#endif // defined(AL_HAS_ROCM) && defined(AL_HAS_NCCL)

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

/** Helper for generating random vectors. */
template <typename T>
struct RandVectorGen {
  template <typename Generator>
  static std::vector<T> gen(size_t count, Generator& g) {
    std::vector<T> v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = gen_random_val<T>(g);
    }
    return v;
  }
};
#ifdef AL_HAS_HALF
// Specialization for half. Standard RNGs do not support half.
template <>
struct RandVectorGen<__half> {
  template <typename Generator>
  static std::vector<__half> gen(size_t count, Generator& g) {
    std::vector<__half> v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = __float2half(gen_random_val<float>(g));
    }
    return v;
  }
};
#endif
#ifdef AL_HAS_BFLOAT
// Specialization for bfloat. Standard RNGs do not support bfloat.
template <>
struct RandVectorGen<al_bfloat16> {
  template <typename Generator>
  static std::vector<al_bfloat16> gen(size_t count, Generator& g) {
    std::vector<al_bfloat16> v(count);
    for (size_t i = 0; i < count; ++i) {
      v[i] = __float2bfloat16(gen_random_val<float>(g));
    }
    return v;
  }
};
#endif

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
    return RandVectorGen<T>::gen(count, rng_gen);
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
 * Get a CommWrapper for the world.
 *
 * This will permute if needed.
 */
template <typename Backend>
CommWrapper<Backend> get_world_wrapper(MPI_Comm world,
                                       cxxopts::ParseResult& parsed_opts) {
  // Permute if needed.
  if (parsed_opts.count("permute")) {
    int comm_size, comm_rank;
    MPI_Comm_size(world, &comm_size);
    MPI_Comm_rank(world, &comm_rank);
    std::vector<int> permutation = parsed_opts["permute"].as<std::vector<int>>();
    if (static_cast<size_t>(comm_size) != permutation.size()) {
      std::cerr << "Invalid --permute, expected " << comm_size << " entries, got "
                << permutation.size() << std::endl;
      std::abort();
    }
    if (permutation[comm_rank] < 0 || permutation[comm_rank] >= comm_size) {
      std::cerr << "Rank " << comm_rank
                << " trying to permute to impossible rank "
                << permutation[comm_rank] << std::endl;
      std::abort();
    }
    MPI_Comm permuted_world;
    MPI_Comm_split(world, 1, permutation[comm_rank], &permuted_world);
    return CommWrapper<Backend>(permuted_world);
  } else {
    return CommWrapper<Backend>(world);
  }
}

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
inline void hang_for_debugging(int hang_rank) {
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
#ifdef AL_HAS_HALF
    {"half", [&]() { functor.template operator()<Backend, __half>(parsed_opts); } },
#endif
#ifdef AL_HAS_BFLOAT
    {"bfloat16", [&]() { functor.template operator()<Backend, al_bfloat16>(parsed_opts); } },
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
    AL_FORCE_CHECK_GPU_NOSYNC(AlGpuGetDeviceCount(&num_gpus));
  }
  return num_gpus;
}

#endif  /** AL_HAS_CUDA */

/**
 * Attempt to identify the global rank from the environment.
 *
 * If this fails, optionally abort or return -1.
 */
inline int get_global_rank(bool required = true) {
  char* env = std::getenv("MV2_COMM_WORLD_RANK");
  if (!env) {
    env = std::getenv("OMPI_COMM_WORLD_RANK");
  }
  if (!env) {
    env = std::getenv("SLURM_PROCID");
  }
  if (!env) {
    env = std::getenv("FLUX_TASK_RANK");
  }
  if (!env) {
    if (required) {
      std::cerr << "Cannot determine global rank" << std::endl;
      std::abort();
    }
    return -1;
  }
  return std::atoi(env);
}

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
    env = std::getenv("FLUX_TASK_LOCAL_ID");
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
  // Flux doesn't have an environment variable for this directly, so we
  // assume an even distribution.
  if (!env) {
    char* flux_size = std::getenv("FLUX_JOB_SIZE");
    if (flux_size) {
      char* flux_nnodes = std::getenv("FLUX_JOB_NNODES");
      if (flux_nnodes) {
        int size = std::atoi(flux_size);
        int nnodes = std::atoi(flux_nnodes);
        return (size + nnodes - 1) / nnodes;
      }
    }
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
  char* env = std::getenv("AL_GPU_ID");
  if (env) {
    const int gpu_id = std::atoi(env);
    AL_FORCE_CHECK_GPU_NOSYNC(AlGpuSetDevice(gpu_id));
    return gpu_id;
  }
  AL_FORCE_CHECK_GPU_NOSYNC(AlGpuSetDevice(local_rank));
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
inline std::vector<size_t> get_sizes(size_t start_size, size_t max_size,
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
inline std::vector<size_t> get_sizes_from_opts(cxxopts::ParseResult& parsed_opts) {
  size_t min_size = parsed_opts["min-size"].as<size_t>();
  size_t max_size = parsed_opts["max-size"].as<size_t>();
  bool odd_sizes = parsed_opts.count("odd-sizes");
  if (parsed_opts.count("size")) {
    min_size = parsed_opts["size"].as<size_t>();
    max_size = min_size;
  }
  return get_sizes(min_size, max_size, odd_sizes);
}

/** Return a human-readable string for size. */
inline std::string human_readable_size(size_t size_) {
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
