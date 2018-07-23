#include <iostream>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

size_t start_size = 1;
size_t max_size = 1<<28;
const size_t num_trials = 10;

template <typename Backend>
void time_allreduce_algo(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    auto recv = get_vector<Backend>(input.size());
    auto in_place_input(input);
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    times.push_back(finish_timer<Backend>(comm));
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(in_place_input.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    in_place_times.push_back(finish_timer<Backend>(comm));
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

std::vector<size_t> get_sizes() {
  std::vector<size_t> sizes = {0};
  for (size_t size = start_size; size <= max_size; size *= 2) {
    sizes.push_back(size);
  }
  return sizes;
}

template <typename Backend>
void do_benchmark(const std::vector<size_t> &sizes) {
  std::vector<typename Backend::algo_type> algos
      = get_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
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
  // Need to set the CUDA device before initializing Aluminum.
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);
  // Add algorithms to test here.

  std::string backend = "MPI";
  if (argc >= 2) {
    backend = argv[1];
  }

  if (argc == 3) {
    max_size = std::atoi(argv[2]);
  }
  if (argc == 4) {
    start_size = std::atoi(argv[2]);
    max_size = std::atoi(argv[3]);
  }
  sizes = get_sizes();
  
  if (backend == "MPI") {
    do_benchmark<Al::MPIBackend>(sizes);
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    do_benchmark<Al::NCCLBackend>(sizes);
#endif    
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    do_benchmark<Al::MPICUDABackend>(sizes);
#endif    
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI";
#ifdef AL_HAS_NCCL
    std::cerr << " | NCCL";
#endif
#ifdef AL_HAS_MPI_CUDA
    std::cerr << " | MPI-CUDA";
#endif
    std::cerr << "]" << std::endl;
    return -1;
  }

  Al::Finalize();
  return 0;
  
}
