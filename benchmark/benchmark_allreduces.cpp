#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"
#ifdef ALUMINUM_HAS_NCCL
#include "test_utils_nccl.hpp"
#endif
#ifdef ALUMINUM_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

const size_t max_size = 1<<30;
const size_t num_trials = 10;

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
void time_allreduce_algo(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = get_time();
    allreduces::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                                   allreduces::ReductionOperator::sum, comm, algo);
    times.push_back(get_time() - start);
    MPI_Barrier(MPI_COMM_WORLD);
    start = get_time();
    allreduces::Allreduce<Backend>(in_place_input.data(), input.size(),
                                   allreduces::ReductionOperator::sum, comm, algo);
    in_place_times.push_back(get_time() - start);
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " regular ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=" << static_cast<int>(algo)
              << " inplace ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void do_benchmark() {
  std::vector<typename Backend::algo_type> algos
      = get_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
  std::vector<size_t> sizes = {0};
  for (size_t size = 1; size <= max_size; size *= 2) {
    sizes.push_back(size);
  }
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    // Benchmark algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      time_allreduce_algo<Backend>(data, comm, algo);        
    }
  }
}

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);
  // Add algorithms to test here.

  std::string backend = "MPI";
  if (argc == 2) {
    backend = argv[1];
  }
  
  if (backend == "MPI") {
    do_benchmark<allreduces::MPIBackend>();
#ifdef ALUMINUM_HAS_NCCL    
  } else if (backend == "NCCL") {
    do_benchmark<allreduces::NCCLBackend>();
#endif    
#ifdef ALUMINUM_HAS_MPI_CUDA    
  } else if (backend == "MPI-CUDA") {
    do_benchmark<allreduces::MPICUDABackend>();
#endif    
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI | NCCL | MPI-CUDA]\n";
    return -1;
  }

  allreduces::Finalize();
  return 0;
  
}
