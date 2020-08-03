#include <iostream>
#include <fstream>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_ht.hpp"
#endif
#ifdef AL_HAS_CUDA_AWARE_MPI
#include "test_utils_cuda_aware_mpi.hpp"
#endif

size_t start_size = 1;
size_t max_size = 1<<28;
const size_t num_trials = 20;

#ifdef AL_HAS_CUDA

template <typename Backend>
void time_allreduce_algo(typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::allreduce_algo_type algo,
                         CollectiveProfile<Backend, typename Backend::allreduce_algo_type>& prof,
                         size_t num_segments, bool mod_seg) {
  auto recv = get_vector<Backend>(input.size());
  auto in_place_input(input);
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    if (trial > 0) {  // Skip warmup.
      prof.add_result(comm, input.size(), algo, false,
                      finish_timer<Backend>(comm), num_segments, mod_seg);
    }
    in_place_input = input;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer<Backend>(comm);
    Al::Allreduce<Backend>(in_place_input.data(), input.size(),
                           Al::ReductionOperator::sum, comm, algo);
    if (trial > 0) {  // Skip warmup.
      prof.add_result(comm, input.size(), algo, true,
                      finish_timer<Backend>(comm), num_segments, mod_seg);
    }
  }
}

template <typename Backend>
void do_benchmark(const std::vector<size_t>& sizes,
                  size_t num_segments, bool mod_segs) {
  std::vector<typename Backend::allreduce_algo_type> algos
      = get_allreduce_algorithms<Backend>();
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm mpi_comm;
  if (mod_segs) {
    MPI_Comm_split(MPI_COMM_WORLD, rank % num_segments, 0, &mpi_comm);
  } else {
    MPI_Comm_split(MPI_COMM_WORLD, rank / (size / num_segments), 0, &mpi_comm);
  }
  typename Backend::comm_type comm(mpi_comm, stream);
  CollectiveProfile<Backend, typename Backend::allreduce_algo_type> prof("allreduce");
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    // Benchmark algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      time_allreduce_algo<Backend>(data, comm, algo, prof, num_segments, mod_segs);
    }
  }
  if (rank == 0) {
    prof.print_result_table();
    std::cout << std::flush;
  }
}

#endif  // AL_HAS_CUDA

std::vector<size_t> load_sizes(const char* filename) {
  std::vector<size_t> sizes;
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Could not load " << filename << std::endl;
    std::abort();
  }
  size_t size;
  while (f >> size) {
    sizes.push_back(size);
  }
  f.close();
  return sizes;
}

int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  if (argc != 5) {
    std::cerr << "Bad arguments" << std::endl;
    return -1;
  }
  std::string backend = argv[1];
  size_t num_segments = std::atoi(argv[2]);
  bool seg_mod = (bool) std::atoi(argv[3]);
  std::vector<size_t> sizes = load_sizes(argv[4]);

#ifndef AL_HAS_CUDA
  // Suppress unused variable warning.
  (void) num_segments;
  (void) seg_mod;
#endif
  
  if (backend == "MPI") {
    // Not supported right now for simplicity.
    //do_benchmark<Al::MPIBackend>(sizes, num_segments, seg_mod);
    std::cerr << "MPI not supported for this benchmark" << std::endl;
    return -1;
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    do_benchmark<Al::NCCLBackend>(sizes, num_segments, seg_mod);
#endif    
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    std::cerr << "Allreduce not supported on MPI-CUDA backend" << std::endl;
    return -1;
#endif
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    do_benchmark<Al::HostTransferBackend>(sizes, num_segments, seg_mod);
#endif
#ifdef AL_HAS_CUDA_AWARE_MPI
  } else if (backend == "CUDA-AWARE-MPI") {
    do_benchmark<Al::CUDAAwareMPIBackend>(sizes, num_segments, seg_mod);
#endif
  } else {
    std::cerr << "Usage: " << argv[0] << " [NCCL | HT | CUDA-AWARE-MPI] #segments modseg sizes" << std::endl;
    return -1;
  }

  Al::Finalize();
  return 0;
  
}
