#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"

const size_t max_size = 1<<20;

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);
  allreduces::MPICommunicator comm;  // Use COMM_WORLD.
  // Compute sizes to test.
  std::vector<size_t> sizes;
  for (size_t size = 1; size <= max_size; size *= 2) {
    for (int i = 0; i < 1000; ++i) {
      sizes.push_back(size);
      // Avoid duplicating 2.
      if (size > 1) {
        sizes.push_back(size + 1);
      }
    }
  }
  std::vector<allreduces::AllreduceRequest> reqs(sizes.size());
  std::vector<std::vector<float>> input_data(sizes.size());
  std::vector<std::vector<float>> expected_data(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    size_t size = sizes[i];
    std::vector<float> data = gen_data(size);
    input_data[i] = data;
    std::vector<float> expected(data);
    MPI_Allreduce(MPI_IN_PLACE, expected.data(), size, MPI_FLOAT, MPI_SUM,
                  MPI_COMM_WORLD);
    expected_data[i] = expected;
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    size_t size = sizes[i];
    allreduces::NonblockingAllreduce<allreduces::MPIBackend>(
      input_data[i].data(), size, allreduces::ReductionOperator::sum, comm,
      reqs[i], allreduces::AllreduceAlgorithm::mpi_recursive_doubling);
  }
  for (size_t i = 0; i < sizes.size(); ++i) {
    size_t size = sizes[i];
    allreduces::Wait(reqs[i]);
    //if (comm.rank() == 0) std::cout << comm.rank() << ": size=" << size << std::endl;
    if (!check_vector(expected_data[i], input_data[i])) {
      std::cout << comm.rank() << ": allreduce doesn't match" << std::endl;
    }
  }
  allreduces::Finalize();
  return 0;
}
