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
    allreduces::AllreduceRequest req;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = get_time();
    allreduces::NonblockingAllreduce<float, Backend>(
        input.data(), recv.data(), input.size(),
        allreduces::ReductionOperator::sum, comm, req, algo);
    allreduces::Wait(req);
    times.push_back(get_time() - start);
    MPI_Barrier(MPI_COMM_WORLD);
    start = get_time();
    allreduces::NonblockingAllreduce<float, Backend>(
        in_place_input.data(), input.size(),
        allreduces::ReductionOperator::sum, comm, req, algo);
    allreduces::Wait(req);
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

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);
  // Add algorithms to test here.
  std::vector<allreduces::AllreduceAlgorithm> algos = {
    allreduces::AllreduceAlgorithm::mpi_passthrough,
    allreduces::AllreduceAlgorithm::mpi_recursive_doubling,
    allreduces::AllreduceAlgorithm::mpi_ring,
    allreduces::AllreduceAlgorithm::mpi_rabenseifner,
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
  allreduces::Finalize();
  return 0;
}
