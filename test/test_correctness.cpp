#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"
#ifdef ALUMINUM_HAS_NCCL
#include "test_utils_nccl.hpp"
#endif
#ifdef ALUMINUM_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

#include <stdlib.h>
#include <math.h>
#include <string>

size_t max_size = 1<<30;

void get_expected_result(std::vector<float>& expected) {
  MPI_Allreduce(MPI_IN_PLACE, expected.data(), expected.size(),
                MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

/**
 * Test allreduce algo on input, check with expected.
 */
template <typename Backend>
void test_allreduce_algo(const typename VectorType<Backend>::type& expected,
                         typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  auto recv = get_vector<Backend>(input.size());
  // Test regular allreduce.
  allreduces::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                                 allreduces::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allreduce does not match" <<
        std::endl;
    std::abort();
  }
  // Test in-place allreduce.
  allreduces::Allreduce<Backend>(input.data(), input.size(),
                                 allreduces::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
  }
}

/**
 * Test non-blocking allreduce algo on input, check with expected.
 */
template <typename Backend>
void test_nb_allreduce_algo(const typename VectorType<Backend>::type& expected,
                            typename VectorType<Backend>::type input,
                            typename Backend::comm_type& comm,
                            typename Backend::algo_type algo) {
  allreduces::AllreduceRequest req;
  auto recv = get_vector<Backend>(input.size());
  // Test regular allreduce.
  allreduces::NonblockingAllreduce<Backend>(input.data(), recv.data(), input.size(),
                                            allreduces::ReductionOperator::sum, comm,
                                            req, algo);
  allreduces::Wait(req);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allreduce does not match" <<
      std::endl;
  }
  // Test in-place allreduce.
  allreduces::NonblockingAllreduce<Backend>(input.data(), input.size(),
                                            allreduces::ReductionOperator::sum, comm,
                                            req, algo);
  allreduces::Wait(req);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
  }
}

template <typename Backend>
void test_correctness() {
  auto algos = get_allreduce_algorithms<Backend>();
  auto nb_algos = get_nb_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
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
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size);
    auto expected(data);
    get_expected_result(expected);
    // Test algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_allreduce_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_nb_allreduce_algo<Backend>(expected, data, comm, algo);
    }
  }
}

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);

  std::string backend = "MPI";
  if (argc >= 2) {
    backend = argv[1];
  }
  if (argc == 3) {
    max_size = std::stoul(argv[2]);
  }

  if (backend == "MPI") {
    test_correctness<allreduces::MPIBackend>();
#ifdef ALUMINUM_HAS_NCCL
  } else if (backend == "NCCL") {
    set_device();
    test_correctness<allreduces::NCCLBackend>();
#endif
#ifdef ALUMINUM_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    set_device();    
    test_correctness<allreduces::MPICUDABackend>();
#endif    
  } else {
    std::cerr << "usage: " << argv[0] << " [MPI | NCCL | MPI-CUDA]\n";
    return -1;
  }


  allreduces::Finalize();
  return 0;
}
