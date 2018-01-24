#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"

#include <stdlib.h>
#include "common.h"


void test_nccl_allreduce(const std::vector<float>& expected,
                         std::vector<float> input,
                         allreduces::NCCLCommunicator& nccl_comm);

const size_t max_size = 1<<30;
const float eps = 1e-4;

bool check_vector(const std::vector<float>& expected,
                  const std::vector<float>& actual) {
  for (size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(expected[i] - actual[i]) > eps) {
      return false;
    }
  }
  return true;
}

/**
 * Test allreduce algo on input, check with expected.
 */
void test_allreduce_algo(const std::vector<float>& expected,
                         std::vector<float> input,
                         allreduces::Communicator& comm,
                         allreduces::AllreduceAlgorithm algo) {
  std::vector<float> recv(input.size());
  // Test regular allreduce.
  allreduces::Allreduce(input.data(), recv.data(), input.size(),
                        allreduces::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allreduce does not match" <<
      std::endl;
  }
  // Test in-place allreduce.
  allreduces::Allreduce(input.data(), input.size(),
                        allreduces::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
  }
}

/**
 * Test non-blocking allreduce algo on input, check with expected.
 */
void test_nb_allreduce_algo(const std::vector<float>& expected,
                            std::vector<float> input,
                            allreduces::Communicator& comm,
                            allreduces::AllreduceAlgorithm algo) {
  allreduces::AllreduceRequest req;
  std::vector<float> recv(input.size());
  // Test regular allreduce.
  allreduces::NonblockingAllreduce(input.data(), recv.data(), input.size(),
                                   allreduces::ReductionOperator::sum, comm,
                                   req, algo);
  allreduces::Wait(req);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allreduce does not match" <<
      std::endl;
  }
  // Test in-place allreduce.
  allreduces::NonblockingAllreduce(input.data(), input.size(),
                                   allreduces::ReductionOperator::sum, comm,
                                   req, algo);
  allreduces::Wait(req);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
  }
}

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);

  int code = 0;
  if(argc == 1)
    code = 0;

  if(argc == 2 && atoi(argv[1]) == 1){
    code = 1;
  }
  else{
    std::cerr << "usage: " << argv[0] << " [0(MPI) | 1(NCCL)]\n";
    return -1;
  }

  if(code == 0){
    // Add algorithms to test here.
    std::vector<allreduces::AllreduceAlgorithm> algos = {
      allreduces::AllreduceAlgorithm::automatic,
      allreduces::AllreduceAlgorithm::mpi_passthrough,
      allreduces::AllreduceAlgorithm::mpi_recursive_doubling,
      allreduces::AllreduceAlgorithm::mpi_ring,
      allreduces::AllreduceAlgorithm::mpi_rabenseifner,
      allreduces::AllreduceAlgorithm::mpi_pe_ring
    };
    std::vector<allreduces::AllreduceAlgorithm> nb_algos = {
      allreduces::AllreduceAlgorithm::automatic,
      allreduces::AllreduceAlgorithm::mpi_passthrough,
      allreduces::AllreduceAlgorithm::mpi_recursive_doubling,
      allreduces::AllreduceAlgorithm::mpi_ring,
      allreduces::AllreduceAlgorithm::mpi_rabenseifner,
      //allreduces::AllreduceAlgorithm::mpi_pe_ring
    };
    allreduces::MPICommunicator comm;  // Use COMM_WORLD.
    // Compute sizes to test.
    std::vector<size_t> sizes = {0};
    for (size_t size = 1; size <= max_size; size *= 2) {
      sizes.push_back(size);
      // Avoid duplicating 2.
      if (size > 1) {
        sizes.push_back(size + 1);
      }
    }
    for (const auto& size : sizes) {
      if (comm.rank() == 0) {
        std::cout << "Testing size " << human_readable_size(size) << std::endl;
      }
      // Compute true value.
      std::vector<float> data = gen_data(size);
      std::vector<float> expected(data);
      MPI_Allreduce(MPI_IN_PLACE, expected.data(), size, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
      // Test algorithms.
      for (auto&& algo : algos) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (comm.rank() == 0) {
          std::cout << " Algo: " << allreduces::allreduce_name(algo) << std::endl;
        }
        test_allreduce_algo(expected, data, comm, algo);
      }
      for (auto&& algo : nb_algos) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (comm.rank() == 0) {
          std::cout << " Algo: NB " << allreduces::allreduce_name(algo) << std::endl;
        }
        test_nb_allreduce_algo(expected, data, comm, algo);
      }
    }
  }
  else{
    allreduces::NCCLCommunicator nccl_comm;  // Use MPI_COMM_WORLD
    // Compute sizes to test.
    std::vector<size_t> sizes = {0};
    for (size_t size = 1; size <= max_size; size *= 2) {
      sizes.push_back(size);
      // Avoid duplicating 2.
      if (size > 1) {
        sizes.push_back(size + 1);
      }
    }

    for (const auto& size : sizes) {
      if (nccl_comm.rank() == 0) {
        std::cout << "Testing size " << human_readable_size(size) << std::endl;
      }
      // Compute true value.
      std::vector<float> data = gen_data(size);
      std::vector<float> expected(data);
      MPI_Allreduce(MPI_IN_PLACE, expected.data(), size, MPI_FLOAT, MPI_SUM,
                    MPI_COMM_WORLD);
        
      MPI_Barrier(MPI_COMM_WORLD);
      test_nccl_allreduce(expected, data, nccl_comm);
    }

  }
    
  allreduces::Finalize();
  return 0;
}




///================
/**
 * Test allreduce algo on input, check with expected.
 */
void test_nccl_allreduce(const std::vector<float>& expected,
                         std::vector<float> input,
                         allreduces::NCCLCommunicator& nccl_comm) {
  std::vector<float> recv(input.size());
  // Test regular allreduce.
  nccl_comm.Allreduce(input.data(), recv.data(), input.size(),
                        allreduces::ReductionOperator::sum);
  if (!check_vector(expected, recv)) {
    std::cout << nccl_comm.rank() << ": regular allreduce does not match" <<
    std::endl;
  }
  /*
  // Test in-place allreduce.
  allreduces::Allreduce(input.data(), input.size(),
                        allreduces::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
  }
*/
}


