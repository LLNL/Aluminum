#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"

const size_t max_size = 1<<22;
const size_t num_trials = 10;
const double sleep_time = 0.005;  // seconds

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

/**
 * Instead of timing purely the runtime, this simulates doing some computation
 * to overlap the communication with. Therefore, the time reported is the
 * non-overlapped communication time.
 * This is only approximate, since the "computation" is done by sleeping.
 */
template <typename Backend>
void time_allreduce_algo(std::vector<float> input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    std::vector<float> recv(input.size());
    std::vector<float> in_place_input(input);
    typename Backend::req_type req = get_request<Backend>();
    MPI_Barrier(MPI_COMM_WORLD);
    double start = get_time();
    allreduces::NonblockingAllreduce<Backend>(
        input.data(), recv.data(), input.size(),
        allreduces::ReductionOperator::sum, comm, req, algo);
    double init_time = get_time();
    std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
    double asleep_time = get_time();
    allreduces::Wait<Backend>(req);
    double end_time = get_time();
    // Because of scheduling issues, use the actual time asleep, not
    // sleep_time, which underestimates.
    double actual_sleep_time = asleep_time - init_time;
    times.push_back(std::max(end_time - start - actual_sleep_time, 0.0));
    MPI_Barrier(MPI_COMM_WORLD);
    start = get_time();
    allreduces::NonblockingAllreduce<Backend>(
        in_place_input.data(), input.size(),
        allreduces::ReductionOperator::sum, comm, req, algo);
    init_time = get_time();
    std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
    asleep_time = get_time();
    allreduces::Wait<Backend>(req);
    end_time = get_time();
    actual_sleep_time = asleep_time - init_time;
    in_place_times.push_back(std::max(end_time - start - actual_sleep_time, 0.0));
    /*if (comm.rank() == 0) {
      std::cout << "size=" << input.size() << " trial=" << trial << " algo=" << static_cast<int>(algo) << " inplace tot=" << (end_time - start) << " init=" << (init_time - start) << " sleep=" << (asleep_time - init_time) << " wait=" << (end_time - asleep_time) << std::endl;
      }*/
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

void time_mpi_baseline(std::vector<float> input,
                       allreduces::Communicator& comm) {
  std::vector<double> times, in_place_times;
  for (size_t trial = 0; trial < num_trials + 1; ++trial) {
    std::vector<float> recv(input.size());
    std::vector<float> in_place_input(input);
    MPI_Request req;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = get_time();
    MPI_Iallreduce(input.data(), recv.data(), input.size(), MPI_FLOAT,
                   MPI_SUM, MPI_COMM_WORLD, &req);
    double init_time = get_time();
    std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
    double asleep_time = get_time();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    double end_time = get_time();
    double actual_sleep_time = asleep_time - init_time;
    times.push_back(std::max(end_time - start - actual_sleep_time, 0.0));
    MPI_Barrier(MPI_COMM_WORLD);
    start = get_time();
    MPI_Iallreduce(MPI_IN_PLACE, in_place_input.data(), input.size(), MPI_FLOAT,
                   MPI_SUM, MPI_COMM_WORLD, &req);
    init_time = get_time();
    std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
    asleep_time = get_time();
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    end_time = get_time();
    actual_sleep_time = asleep_time - init_time;
    in_place_times.push_back(std::max(end_time - start - actual_sleep_time, 0.0));
  }
  // Delete warmup trial.
  times.erase(times.begin());
  in_place_times.erase(in_place_times.begin());
  if (comm.rank() == 0) {
    std::cout << "size=" << input.size() << " algo=-1 regular ";
    print_stats(times);
    std::cout << "size=" << input.size() << " algo=-1 inplace ";
    print_stats(in_place_times);
  }
}

template <typename Backend>
void do_benchmark() {
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
  //std::vector<size_t> sizes = {64, 128};
  for (const auto& size : sizes) {
    auto data = gen_data<Backend>(size);
    // Benchmark algorithms.
    time_mpi_baseline(data, comm);
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      time_allreduce_algo<Backend>(data, comm, algo);
    }
  }
}

int main(int argc, char *argv[]) {
  allreduces::Initialize(argc, argv);
  std::string backend = "MPI";
  if (argc == 2) {
    backend = argv[1];
  }
  
  if (backend == "MPI") {
    do_benchmark<allreduces::MPIBackend>();
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI | NCCL]\n";
    return -1;
  }

  allreduces::Finalize();
  return 0;
}
