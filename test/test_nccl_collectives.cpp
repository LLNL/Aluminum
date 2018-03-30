#include <iostream>
#include "allreduce.hpp"
#include "test_utils.hpp"
#ifdef ALUMINUM_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef ALUMINUM_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

#include <stdlib.h>
#include <math.h>
#include <string>

size_t max_size = 1<<30;

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
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(input.size());

  // Test nonblocking allreduce.
  allreduces::NonblockingAllreduce<Backend>(input.data(), recv.data(), input.size(),
                                            allreduces::ReductionOperator::sum, comm,
                                            req, algo);

  allreduces::Wait<Backend>(req);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": non-blocking allreduce does not match" <<
      std::endl;
  }

  allreduces::NonblockingAllreduce<Backend>(input.data(), input.size(),
                                            allreduces::ReductionOperator::sum, comm,
                                            req, algo);
  allreduces::Wait<Backend>(req);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place non-blocking allreduce does not match" <<
      std::endl;
  }
}

template <typename Backend>
void test_reduce_algo(const typename VectorType<Backend>::type& expected,
                      typename VectorType<Backend>::type input,
                      typename Backend::comm_type& comm,
                      typename Backend::algo_type algo) {
  auto recv = get_vector<Backend>(input.size());
  
  allreduces::Reduce<Backend>(input.data(), recv.data(), input.size(), allreduces::ReductionOperator::sum, 0, comm, algo);
  if(comm.rank() == 0){
    if (!check_vector(expected, recv)) {
      std::cout << comm.rank() << ": regular reduce does not match" <<
          std::endl;
      std::abort();
    }
  }

  allreduces::Reduce<Backend>(input.data(), input.size(), allreduces::ReductionOperator::sum, 0, comm, algo);
  if(comm.rank() == 0){
    if (!check_vector(expected, recv)) {
      std::cout << comm.rank() << ": in-place reduce does not match" <<
          std::endl;
      std::abort();
    }
  }
}


template <typename Backend>
void test_nb_reduce_algo(const typename VectorType<Backend>::type& expected,
                         typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(input.size());
  // Test regular reduce.
  allreduces::NonblockingReduce<Backend>(input.data(), recv.data(), input.size(), allreduces::ReductionOperator::sum, 0, comm, req, algo);

  allreduces::Wait<Backend>(req);
  if(comm.rank() == 0){
    if (!check_vector(expected, recv)) {
      std::cout << comm.rank() << ": non-blocking regular reduce does not match" <<
        std::endl;

      std::abort();
    }
  }

  // Test in-place reduce.
  allreduces::NonblockingReduce<Backend>(input.data(), input.size(),
                                            allreduces::ReductionOperator::sum, 0, comm,
                                            req, algo);
  allreduces::Wait<Backend>(req);
  if(comm.rank() == 0){
    if (!check_vector(expected, input)) {
      std::cout << comm.rank() << ": non-blocking in-place reduce does not match" <<
        std::endl;
      std::abort();
    }
  }
}


template <typename Backend>
void test_reduce_scatter_algo(const typename VectorType<Backend>::type& expected,
                         typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {

  auto recv = get_vector<Backend>(expected.size());
  std::vector<size_t> recv_count(comm.size());
  for(int i=0; i<comm.size(); i++){
    recv_count[i] = expected.size();
  }

  allreduces::Reduce_scatter<Backend>(input.data(), recv.data(), recv_count.data(), allreduces::ReductionOperator::sum, comm, algo);

  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular reduce_scatter does not match" <<
        std::endl;
    std::abort();
  }

  auto input_copy (input);
  allreduces::Reduce_scatter<Backend>(input_copy.data(), recv_count.data(), allreduces::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, input_copy)) {
    std::cout << comm.rank() << ": in-place reduce_scatter does not match" <<
      std::endl;
    std::abort();
  }
}


template <typename Backend>
void test_nb_reduce_scatter_algo(const typename VectorType<Backend>::type& expected,
                         typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {

  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(expected.size());
  std::vector<size_t> recv_count(comm.size());
  for(int i=0; i<comm.size(); i++){
    recv_count[i] = expected.size();
  }

  /// Test regular reduce_scatter
  allreduces::NonblockingReduce_scatter<Backend>(input.data(), recv.data(), recv_count.data(), allreduces::ReductionOperator::sum, comm, req, algo);
  allreduces::Wait<Backend>(req);

  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": non-blocking regular reduce_scatter does not match" <<
        std::endl;
    std::abort();
  }

  /// Test in-place reduce_scatter
  auto input_copy (input);
  allreduces::NonblockingReduce_scatter<Backend>(input_copy.data(), recv_count.data(), allreduces::ReductionOperator::sum, comm, req,  algo);
  if (!check_vector(expected, input_copy)) {
    std::cout << comm.rank() << ": non-blocking in-place reduce_scatter does not match" <<
      std::endl;
    std::abort();
  }
}

template <typename Backend>
void test_allgather_algo(const typename VectorType<Backend>::type& expected,
                         typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::algo_type algo) {
  auto recv = get_vector<Backend>(expected.size());
  allreduces::Allgather<Backend>(input.data(), recv.data(), input.size(), comm, algo);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allgather does not match" <<
        std::endl;
    std::abort();
  }
  
  // Copy input to recv
  recv.move(input);
  allreduces::Allgather<Backend>(recv.data(), input.size(), comm, algo);
  if (!check_vector(expected, recv )) {
    std::cout << comm.rank() << ": in-place allgather does not match" <<
      std::endl;
    std::abort();
  }
}

template <typename Backend>
void test_nb_allgather_algo(const typename VectorType<Backend>::type& expected,
                            typename VectorType<Backend>::type input,
                            typename Backend::comm_type& comm,
                            typename Backend::algo_type algo) {
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(expected.size());
  allreduces::NonblockingAllgather<Backend>(input.data(), recv.data(), input.size(), comm, req, algo);

  allreduces::Wait<Backend>(req);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": non-blocking regular allgather does not match" <<
      std::endl;
  }

  recv.move(input);
  allreduces::NonblockingAllgather<Backend>(recv.data(), input.size(), comm, req, algo);
  allreduces::Wait<Backend>(req);
  if (!check_vector(expected, recv )) {
    std::cout << comm.rank() << ": non-blocking in-place allgather does not match" <<
      std::endl;
  }
}

template <typename Backend>
void test_bcast_algo(const typename VectorType<Backend>::type& expected,
                     typename VectorType<Backend>::type input,
                     typename Backend::comm_type& comm,
                     typename Backend::algo_type algo) {

  auto input_copy(input);
  allreduces::Bcast<Backend>(input_copy.data(), input_copy.size(),  0, comm, algo);
  if (!check_vector(expected, input_copy)) {
    std::cout << comm.rank() << ": regular bcast does not match" <<
        std::endl;
    std::abort();
  }

  /// Bcast is by default an in-place collective, so no need for a separate in-place test
}


template <typename Backend>
void test_nb_bcast_algo(const typename VectorType<Backend>::type& expected,
                        typename VectorType<Backend>::type input,
                        typename Backend::comm_type& comm,
                        typename Backend::algo_type algo) {

  typename Backend::req_type req = get_request<Backend>();
  auto input_copy(input);

  allreduces::NonblockingBcast<Backend>(input_copy.data(), input_copy.size(),  0, comm, req, algo);
  if (!check_vector(expected, input_copy)) {
    std::cout << comm.rank() << ": in-place bcast does not match" <<
        std::endl;
    std::abort();
  }
  
  /// Bcast is by default an in-place collective, so no need for a separate in-place test
}

template <typename Backend>
void test_correctness() {
  auto algos = get_allreduce_algorithms<Backend>();
  auto nb_algos = get_nb_allreduce_algorithms<Backend>();
  typename Backend::comm_type comm;  // Use COMM_WORLD.
  // Compute sizes to test.
  std::vector<size_t> sizes;
  for (size_t size = 1; size <= max_size; size *= 2) {
    sizes.push_back(size);
    /// Avoid duplicating 2.
    if (size > 1) {
      sizes.push_back(size + 1);
    }
  }

  int num_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /// Allreduce testing
  if(comm.rank() == 0){
    std::cout << "======================================" << std::endl;
    std::cout << "Testing NCCL Allreduce collective " << std::endl;
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    /// Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size);
    auto expected(data);
    get_expected_nccl_result_allreduce(expected);

    /// Test algorithms.
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

  /// Reduce Testing
  if(comm.rank() == 0){
    std::cout << "======================================" << std::endl;
    std::cout << "Testing NCCL Reduce collective " << std::endl;
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    /// Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size);
    auto expected(data);
    get_expected_nccl_result_reduce(expected);

    /// Test algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_reduce_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_nb_reduce_algo<Backend>(expected, data, comm, algo);
    }
  }

  /// Allgather Testing
  if(comm.rank() == 0){
    std::cout << "======================================" << std::endl;
    std::cout << "Testing NCCL Allgather collective " << std::endl;
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    // Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size);
    auto expected = get_vector<Backend>(size*num_procs);

    get_expected_nccl_result_allgather(data, expected);

    // Test algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_allgather_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_nb_allgather_algo<Backend>(expected, data, comm, algo);
    }
  }
  
  /// Reduce_scatter Testing
  if(comm.rank() == 0){
    std::cout << "======================================" << std::endl;
    std::cout << "Testing NCCL Reduce_scatter collective " << std::endl;
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    // Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size*num_procs);
    auto expected = get_vector<Backend>(size);

    get_expected_nccl_result_reduce_scatter(data, expected);

    // Test algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_reduce_scatter_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_nb_reduce_scatter_algo<Backend>(expected, data, comm, algo);
    }
  }
  
  /// Bcast Testing
  if(comm.rank() == 0){
    std::cout << "======================================" << std::endl;
    std::cout << "Testing NCCL Bcast collective " << std::endl;
  }
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    // Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size);
    auto expected(data);

    get_expected_nccl_result_bcast(expected);

    // Test algorithms.
    for (auto&& algo : algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_bcast_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << allreduces::allreduce_name(algo) << std::endl;
      }
      test_nb_bcast_algo<Backend>(expected, data, comm, algo);
    }
  }
}

int main(int argc, char** argv) {
  allreduces::Initialize(argc, argv);

  std::string backend = "";
  max_size = std::stoul(argv[1]);

#ifdef ALUMINUM_HAS_NCCL
  if (backend == "NCCL") {
    set_device();
    test_correctness<allreduces::NCCLBackend>();
  }
#endif
  if (backend == "") {
    std::cerr << "usage: " << argv[0] << " [NCCL] \n";
    return -1;
  }

  allreduces::Finalize();
  return 0;
}
  
