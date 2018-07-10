#include <iostream>
#include <string>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

size_t max_size = 1<<20;
const size_t num_concurrent = 1024;

void get_expected_result(std::vector<float>& expected) {
  MPI_Allreduce(MPI_IN_PLACE, expected.data(), expected.size(), MPI_FLOAT,
                MPI_SUM, MPI_COMM_WORLD);
}

template <typename Backend>
void test_multiple_nballreduces() {
  auto algos = get_nb_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
  std::vector<size_t> sizes = {0};
  for (size_t size = 1; size <= max_size; size *= 2) {
    sizes.push_back(size);
    if (size > 1) {
      sizes.push_back(size + 1);
    }
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    for (auto&& algo : algos) {
      std::vector<typename Backend::req_type> reqs(num_concurrent);
      std::vector<typename VectorType<Backend>::type> input_data;
      std::vector<typename VectorType<Backend>::type> expected_results;
      for (size_t i = 0; i < num_concurrent; ++i) {
        input_data.push_back(std::move(gen_data<Backend>(size)));
        expected_results.push_back(std::move(typename VectorType<Backend>::type(input_data[i])));
        get_expected_result(expected_results[i]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << Al::allreduce_name(algo) << std::endl;
      }
      // Start each allreduce.
      for (size_t i = 0; i < num_concurrent; ++i) {
        Al::NonblockingAllreduce<Backend>(input_data[i].data(),
                                          input_data[i].size(),
                                          Al::ReductionOperator::sum,
                                          comm,
                                          reqs[i],
                                          algo);
      }
      // Complete and check them.
      for (size_t i = 0; i < num_concurrent; ++i) {
        Al::Wait<Backend>(reqs[i]);
        if (!check_vector(expected_results[i], input_data[i])) {
          std::cout << comm.rank() << ": allreduce does not match" << std::endl;
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
      }
    }
  }
}

int main(int argc, char** argv) {
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI";
  if (argc >= 2) {
    backend = argv[1];
  }
  if (argc == 3) {
    max_size = std::stoul(argv[2]);
  }

  if (backend == "MPI") {
    test_multiple_nballreduces<Al::MPIBackend>();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    test_multiple_nballreduces<Al::NCCLBackend>();
#endif
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    test_multiple_nballreduces<Al::MPICUDABackend>();
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
