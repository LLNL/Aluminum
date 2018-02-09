#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"

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
void time_allreduce_algo(std::vector<float> input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    std::vector<float> recv(input.size());
    std::vector<float> in_place_input(input);
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

void time_nccl_allreduce(std::vector<float> input,
                         allreduces::NCCLCommunicator& comm) {

  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    float *sbuffer;
    float *rbuffer;
    size_t len = input.size() * sizeof(float);

    CUDACHECK(cudaMalloc((void **)&sbuffer, len));
    CUDACHECK(cudaMalloc((void **)&rbuffer, len));
    CUDACHECK(cudaMemcpy(sbuffer, &input[0], len, cudaMemcpyHostToDevice));


    MPI_Barrier(MPI_COMM_WORLD);
    double start = get_time();
    allreduces::NCCLAllreduce(sbuffer, rbuffer, input.size(),
                          allreduces::ReductionOperator::sum, comm);
    times.push_back(get_time() - start);
    MPI_Barrier(MPI_COMM_WORLD);
    start = get_time();
    allreduces::Allreduce(sbuffer, sbuffer, input.size(),
                          allreduces::ReductionOperator::sum, comm);
    in_place_times.push_back(get_time() - start);

    CUDACHECK(cudaFree(sbuffer));
    CUDACHECK(cudaFree(rbuffer));
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "NCCL: size=" << input.size() << " regular ";
    print_stats(times);
    std::cout << "NCCL: size=" << input.size() << " inplace ";
    print_stats(in_place_times);
  }
}

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);
  // Add algorithms to test here.

  int code = 0;
  if(argc == 1){
    code = 0;
  }
  else if(argc == 2) {
    code = atoi(argv[1]);
    if(code != 0 && code != 1){
      std::cerr << "usage: " << argv[0] << " [0(MPI) | 1(NCCL)]\n";
      return -1;
    }
  }
  else{
    std::cerr << "usage: " << argv[0] << " [0(MPI) | 1(NCCL)]\n";
    return -1;
  }


  if(code == 0){
    std::vector<allreduces::AllreduceAlgorithm> algos = {
      allreduces::AllreduceAlgorithm::mpi_passthrough,
      allreduces::AllreduceAlgorithm::mpi_recursive_doubling,
      allreduces::AllreduceAlgorithm::mpi_ring,
      allreduces::AllreduceAlgorithm::mpi_rabenseifner,
      allreduces::AllreduceAlgorithm::mpi_pe_ring
    };
    allreduces::MPICommunicator comm;  // Use COMM_WORLD.
    std::vector<size_t> sizes = {0};
    for (size_t size = 1; size <= max_size; size *= 2) {
      sizes.push_back(size);
    }
    for (const auto& size : sizes) {
      std::vector<float> data = gen_data(size);
      // Benchmark algorithms.
      for (auto&& algo : algos) {
        MPI_Barrier(MPI_COMM_WORLD);
        time_allreduce_algo<allreduces::MPIBackend>(data, comm, algo);        
      }
    }
  }
  else{
    allreduces::NCCLCommunicator nccl_comm;  
    std::vector<size_t> sizes = {0};
    for (size_t size = 1; size <= max_size; size *= 2) {
      sizes.push_back(size);
    }
      
    for (const auto& size : sizes) {
      std::vector<float> data = gen_data(size);
      MPI_Barrier(MPI_COMM_WORLD);
      time_nccl_allreduce(data, nccl_comm);
    }
  }

  allreduces::Finalize();
  return 0;
  
}
