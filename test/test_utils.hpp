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
std::vector<typename Backend::algo_type> get_allreduce_algorithms() {
  std::vector<typename Backend::algo_type> algos = {
    Backend::algo_type::automatic};
  return algos;
}
 
template <>
std::vector<Al::MPIBackend::algo_type>
get_allreduce_algorithms<Al::MPIBackend>() {  
   std::vector<Al::AllreduceAlgorithm> algos = {
     Al::AllreduceAlgorithm::automatic,
     Al::AllreduceAlgorithm::mpi_passthrough,
     Al::AllreduceAlgorithm::mpi_recursive_doubling,
     Al::AllreduceAlgorithm::mpi_ring,
     Al::AllreduceAlgorithm::mpi_rabenseifner,
     Al::AllreduceAlgorithm::mpi_pe_ring,
     Al::AllreduceAlgorithm::mpi_biring
  };
  return algos;
}

template <typename Backend>
std::vector<typename Backend::algo_type> get_nb_allreduce_algorithms() {
  std::vector<typename Backend::algo_type> algos = {
    Backend::algo_type::automatic};
  return algos;
}
 
template <>
std::vector<Al::MPIBackend::algo_type>
get_nb_allreduce_algorithms<Al::MPIBackend>() {  
  std::vector<Al::AllreduceAlgorithm> algos = {
    Al::AllreduceAlgorithm::automatic,
    Al::AllreduceAlgorithm::mpi_passthrough,
    Al::AllreduceAlgorithm::mpi_recursive_doubling,
    Al::AllreduceAlgorithm::mpi_ring,
    Al::AllreduceAlgorithm::mpi_rabenseifner
    //Al::AllreduceAlgorithm::mpi_pe_ring
  };
  return algos;
}

#define eps (1e-4)

bool check_vector(const std::vector<float>& expected,
                  const std::vector<float>& actual) {
  bool match = true;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  for (size_t i = 0; i < expected.size(); ++i) {
    float e = expected[i];

    if (std::abs(e - actual[i]) > eps) {
#ifdef AL_DEBUG
      std::stringstream ss;
      ss << "[" << rank << "] @" << i << " Expected: " << e
                << ", Actual: " << actual[i] << "\n";
      std::cerr << ss.str();
      match = false;
      return false;
#else
      return false;
#endif
    }
  }
  return match;
}

void print_stats(std::vector<double>& times) {
  double sum = std::accumulate(times.begin(), times.end(), 0.0);
  double mean = sum / times.size();
  std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
  double median = times[times.size() / 2];
  auto minmax = std::minmax_element(times.begin(), times.end());
  double min = *(minmax.first);
  double max = *(minmax.second);
  std::cout << "mean=" << mean << " median=" << median << " min=" << min <<
    " max=" << max << std::endl;
}

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
