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

#include <vector>
#include <random>
#include <chrono>
#include <sstream>

#include "Al.hpp"

namespace {
static std::mt19937 rng_gen;
static bool rng_seeded = false;
}

template <typename Backend>
struct VectorType {
  using type = std::vector<float>;
};

template <typename Backend>
typename VectorType<Backend>::type get_vector(size_t count) {
  return typename VectorType<Backend>::type(count);
}

/** Parse input arguments. */
void parse_args(int argc, char** argv,
                std::string& backend, size_t& start_size, size_t& max_size) {
  if (argc == 1) {
    backend = "MPI";
    return;
  } else {
    backend = argv[1];
    if (argc == 3) {
      start_size = std::stoul(argv[2]);
      max_size = start_size;
    } else if (argc == 4) {
      start_size = std::stoul(argv[2]);
      max_size = std::stoul(argv[3]);
    } else if (argc > 5) {
      std::cerr << "Unexpected argument." << std::endl;
      std::abort();
    }
  }
  if (backend != "MPI"
#ifdef AL_HAS_NCCL
      && backend != "NCCL"
#endif
#ifdef AL_HAS_MPI_CUDA
      && backend != "MPI-CUDA"
#endif
    ) {
    std::cerr << "Usage: " << argv[0] << " [MPI"
#ifdef AL_HAS_NCCL
              << " | NCCL"
#endif
#ifdef AL_HAS_MPI_CUDA
              << " | MPI-CUDA"
#endif
              << "] [start size] [max size]"
              << std::endl;
    std::abort();
  }
}

/**
 * Return every size to test between start_size and max_size (inclusive).
 * If odds is true, generate odd-numbered values too.
 */
std::vector<size_t> get_sizes(size_t start_size, size_t max_size,
                              bool odds = false) {
  std::vector<size_t> sizes;
  if (start_size == 0) {
    sizes.push_back(0);
  }
  for (size_t size = start_size; size <= max_size; size *= 2) {
    sizes.push_back(size);
    if (odds && size > 1) {
      sizes.push_back(size + 1);
    }
  }
  return sizes;
}

/** Generate random data of length count. */
template <typename Backend=Al::MPIBackend>
typename VectorType<Backend>::type gen_data(size_t count);

template <>
typename VectorType<Al::MPIBackend>::type
gen_data<Al::MPIBackend>(size_t count) {
  if (!rng_seeded) {
    int flag;
    MPI_Initialized(&flag);
    if (flag) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      rng_gen.seed(rank);
      rng_seeded = true;
    }
  }
  std::uniform_real_distribution<float> rng;
  std::vector<float> v(count);
  for (size_t i = 0; i < count; ++i) {
    v[i] = rng(rng_gen);
  }
  return v;
}

template <typename Backend=Al::MPIBackend>
typename VectorType<Backend>::type create_data(size_t count);

template <>
typename VectorType<Al::MPIBackend>::type
create_data<Al::MPIBackend>(size_t count) {
  std::vector<float> v(count);
  for (size_t i = 0; i < count; ++i) {
    v[i] = 0.0;
  }
  return v;
}

/** Get current time. */
inline double get_time() {                                                      
  using namespace std::chrono;                                                  
  return duration_cast<duration<double>>(                                       
    steady_clock::now().time_since_epoch()).count();                            
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

template <typename Backend>
std::vector<typename Backend::allreduce_algo_type> get_allreduce_algorithms() {
  std::vector<typename Backend::allreduce_algo_type> algos = {
    Backend::allreduce_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::reduce_algo_type> get_reduce_algorithms() {
  std::vector<typename Backend::reduce_algo_type> algos = {
    Backend::reduce_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::reduce_scatter_algo_type> get_reduce_scatter_algorithms() {
  std::vector<typename Backend::reduce_scatter_algo_type> algos = {
    Backend::reduce_scatter_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::allgather_algo_type> get_allgather_algorithms() {
  std::vector<typename Backend::allgather_algo_type> algos = {
    Backend::allgather_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::bcast_algo_type> get_bcast_algorithms() {
  std::vector<typename Backend::bcast_algo_type> algos = {
    Backend::bcast_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::alltoall_algo_type> get_alltoall_algorithms() {
  std::vector<typename Backend::alltoall_algo_type> algos = {
    Backend::alltoall_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::gather_algo_type> get_gather_algorithms() {
  std::vector<typename Backend::gather_algo_type> algos = {
    Backend::gather_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::scatter_algo_type> get_scatter_algorithms() {
  std::vector<typename Backend::scatter_algo_type> algos = {
    Backend::scatter_algo_type::automatic};
  return algos;
}

template <>
std::vector<Al::MPIBackend::allreduce_algo_type>
get_allreduce_algorithms<Al::MPIBackend>() {  
   std::vector<Al::MPIAllreduceAlgorithm> algos = {
     Al::MPIAllreduceAlgorithm::automatic,
     Al::MPIAllreduceAlgorithm::mpi_passthrough,
     Al::MPIAllreduceAlgorithm::mpi_recursive_doubling,
     Al::MPIAllreduceAlgorithm::mpi_ring,
     Al::MPIAllreduceAlgorithm::mpi_rabenseifner,
     Al::MPIAllreduceAlgorithm::mpi_pe_ring,
     Al::MPIAllreduceAlgorithm::mpi_biring
  };
  return algos;
}

template <typename Backend>
std::vector<typename Backend::allreduce_algo_type> get_nb_allreduce_algorithms() {
  std::vector<typename Backend::allreduce_algo_type> algos = {
    Backend::allreduce_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::reduce_algo_type> get_nb_reduce_algorithms() {
  std::vector<typename Backend::reduce_algo_type> algos = {
    Backend::reduce_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::reduce_scatter_algo_type> get_nb_reduce_scatter_algorithms() {
  std::vector<typename Backend::reduce_scatter_algo_type> algos = {
    Backend::reduce_scatter_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::allgather_algo_type> get_nb_allgather_algorithms() {
  std::vector<typename Backend::allgather_algo_type> algos = {
    Backend::allgather_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::bcast_algo_type> get_nb_bcast_algorithms() {
  std::vector<typename Backend::bcast_algo_type> algos = {
    Backend::bcast_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::alltoall_algo_type> get_nb_alltoall_algorithms() {
  std::vector<typename Backend::alltoall_algo_type> algos = {
    Backend::alltoall_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::gather_algo_type> get_nb_gather_algorithms() {
  std::vector<typename Backend::gather_algo_type> algos = {
    Backend::gather_algo_type::automatic};
  return algos;
}

template <typename Backend>
std::vector<typename Backend::scatter_algo_type> get_nb_scatter_algorithms() {
  std::vector<typename Backend::scatter_algo_type> algos = {
    Backend::scatter_algo_type::automatic};
  return algos;
}
 
template <>
std::vector<Al::MPIBackend::allreduce_algo_type>
get_nb_allreduce_algorithms<Al::MPIBackend>() {  
  std::vector<Al::MPIAllreduceAlgorithm> algos = {
    Al::MPIAllreduceAlgorithm::automatic,
    Al::MPIAllreduceAlgorithm::mpi_passthrough,
    Al::MPIAllreduceAlgorithm::mpi_recursive_doubling,
    Al::MPIAllreduceAlgorithm::mpi_ring,
    Al::MPIAllreduceAlgorithm::mpi_rabenseifner
    //Al::MPIAllreduceAlgorithm::mpi_pe_ring
  };
  return algos;
}

#define eps (1e-4)

bool check_vector(const std::vector<float>& expected,
                  const std::vector<float>& actual,
                  size_t start = 0,
                  size_t end = std::numeric_limits<size_t>::max()) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (end == std::numeric_limits<size_t>::max()) {
    end = expected.size();
  }
  for (size_t i = start; i < end; ++i) {
    float e = expected[i];
    if (std::abs(e - actual[i]) > eps) {
#ifdef AL_DEBUG
      std::stringstream ss;
      ss << "[" << rank << "] @" << i << " Expected: " << e
                << ", Actual: " << actual[i] << "\n";
      // Helpful for debugging to print out small vectors completely.
      if (expected.size() < 128) {
        ss << "[" << rank << "] expected: ";
        for (const auto& v : expected) ss << v << " ";
        ss << "actual: ";
        for (const auto& v : actual) ss << v << " ";
        ss << "\n";
      }
      std::cerr << ss.str();
#endif
      return false;
    }
  }
  return true;
}

// Stores statistics related to a set of measurements.
struct MeasurementStats {
  // This is not const or by reference so we can use nth_element.
  MeasurementStats(std::vector<double> times) {
    const double sum = std::accumulate(times.begin(), times.end(), 0.0);
    mean = sum / times.size();
    if (times.size() > 1) {
      double sqsum = 0.0;
      for (const auto& t : times) sqsum += (t - mean) * (t - mean);
      stdev = std::sqrt(1.0 / (times.size() - 1) * sqsum);
    }
    std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
    median = times[times.size() / 2];
    auto minmax = std::minmax_element(times.begin(), times.end());
    min = *(minmax.first);
    max = *(minmax.second);
  }
  double mean = 0.0;
  double median = 0.0;
  double min = 0.0;
  double max = 0.0;
  double stdev = 0.0;
};

void print_stats(std::vector<double>& times) {
  MeasurementStats stats(times);
  std::cout << "mean=" << stats.mean << " median=" << stats.median <<
    " min=" << stats.min << " max=" << stats.max << " stdev=" << stats.stdev <<
    std::endl;
}

// Stores runs for profiling a collective.
template <typename Backend>
struct CollectiveProfile {
  CollectiveProfile(std::string coll_name_) : coll_name(coll_name_) {}
  void add_result(const typename Backend::comm_type& comm,
                  size_t size, typename Backend::allreduce_algo_type algo,
                  bool inplace, double t) {
    results.emplace_back(std::make_tuple(
                           comm.size(), comm.rank(), size, algo, inplace, t));
  }
  void print_result_table() {
    // Print header.
    std::cout << "Backend Collective CommSize CommRank CollSize Algo InPlace Time" << std::endl;
    for (const auto& result : results) {
      std::cout << Backend::Name() << " "
                << coll_name << " "
                << std::get<0>(result) << " "
                << std::get<1>(result) << " "
                << std::get<2>(result) << " "
                << Al::algorithm_name(std::get<3>(result)) << " "
                << std::get<4>(result) << " "
                << std::get<5>(result) << "\n";
    }
    std::flush(std::cout);
  }
  std::string coll_name;
  // Communicator size, rank, collective size, algorithm, inplace time.
  // TODO: Generalize beyond allreduce algos.
  std::vector<std::tuple<int, int, size_t,
                         typename Backend::allreduce_algo_type, bool, double>>
                   results;
};

template <typename Backend>
void start_timer(typename Backend::comm_type& comm);

template <typename Backend>
double finish_timer(typename Backend::comm_type& comm);

inline double& get_cur_time() {
  static double t = 0.0;
  return t;
}

template <>
inline void start_timer<Al::MPIBackend>(typename Al::MPIBackend::comm_type&) {
  double& t = get_cur_time();
  t = get_time();
}

template <>
inline double finish_timer<Al::MPIBackend>(typename Al::MPIBackend::comm_type&) {
  double& t = get_cur_time();
  return get_time() - t;
}

template <typename Backend>
typename Backend::req_type get_request();

template <>
inline typename Al::MPIBackend::req_type
get_request<Al::MPIBackend>() {
  return Al::MPIBackend::null_req;
}

template <typename Backend>
typename Backend::comm_type get_comm_with_stream(MPI_Comm c);

template <typename Backend>
void free_comm_with_stream(typename Backend::comm_type& c);

template <>
inline typename Al::MPIBackend::comm_type get_comm_with_stream<Al::MPIBackend>(
  MPI_Comm c) {
  return Al::MPIBackend::comm_type(c);
}

template <>
inline void free_comm_with_stream<Al::MPIBackend>(
  typename Al::MPIBackend::comm_type&) {}

void get_expected_allreduce_result(std::vector<float>& expected) {
  MPI_Allreduce(MPI_IN_PLACE, expected.data(), expected.size(),
                MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void get_expected_reduce_scatter_result(std::vector<float>& expected) {
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Reduce_scatter_block(MPI_IN_PLACE, expected.data(),
                           expected.size() / nprocs,
                           MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

void get_expected_allgather_result(std::vector<float>& input,
                                   std::vector<float>& expected) {
  MPI_Allgather(input.data(), input.size(), MPI_FLOAT,
                expected.data(), input.size(), MPI_FLOAT,
                MPI_COMM_WORLD);
}

void get_expected_allgather_inplace_result(std::vector<float>& expected) {
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Allgather(MPI_IN_PLACE, expected.size() / nprocs, MPI_FLOAT,
                expected.data(), expected.size() / nprocs, MPI_FLOAT,
                MPI_COMM_WORLD);
}

void get_expected_alltoall_result(std::vector<float>& expected) {
  int nprocs;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Alltoall(MPI_IN_PLACE, expected.size() / nprocs, MPI_FLOAT,
               expected.data(), expected.size() / nprocs, MPI_FLOAT,
               MPI_COMM_WORLD);
}

void get_expected_reduce_result(std::vector<float>& expected) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, expected.data(), expected.size(),
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  } else {
    MPI_Reduce(expected.data(), expected.data(), expected.size(),
               MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  }
}

void get_expected_bcast_result(std::vector<float>& expected) {
  MPI_Bcast(expected.data(), expected.size(), MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void get_expected_gather_result(std::vector<float>& expected) {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (rank == 0) {
    MPI_Gather(MPI_IN_PLACE, expected.size() / nprocs, MPI_FLOAT,
               expected.data(), expected.size() / nprocs, MPI_FLOAT,
               0, MPI_COMM_WORLD);
  } else {
    MPI_Gather(expected.data(), expected.size() / nprocs, MPI_FLOAT,
               expected.data(), expected.size() / nprocs, MPI_FLOAT,
               0, MPI_COMM_WORLD);
  }
}

void get_expected_scatter_result(std::vector<float>& expected) {
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (rank == 0) {
    MPI_Scatter(expected.data(), expected.size() / nprocs, MPI_FLOAT,
                MPI_IN_PLACE, expected.size() / nprocs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
  } else {
    MPI_Scatter(expected.data(), expected.size() / nprocs, MPI_FLOAT,
                expected.data(), expected.size() / nprocs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
  }
}
