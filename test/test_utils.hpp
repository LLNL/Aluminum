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
/*
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (!rng_seeded) {
    int flag;
    MPI_Initialized(&flag);
    if (flag) {
      rng_gen.seed(rank);
    }
  }
  std::uniform_real_distribution<float> rng;
  std::vector<float> v(count);
  for (size_t i = 0; i < count; ++i) {
    v[i] = (float) (count*rank+i);
  }
  return v;
*/

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
    Al::AllreduceAlgorithm::mpi_pe_ring
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
    Al::AllreduceAlgorithm::mpi_rabenseifner,
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
#ifdef ALUMINUM_DEBUG
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

template <typename Backend>
typename Backend::req_type get_request();

template <>
inline typename Al::MPIBackend::req_type
get_request<Al::MPIBackend>() {
  int req = 0;
  return req;
}
