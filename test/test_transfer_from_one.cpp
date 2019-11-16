#include <iostream>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_MPI_CUDA
#include "test_utils_mpi_cuda.hpp"
#endif

#include <stdlib.h>
#include <math.h>
#include <string>
#include <sstream>

size_t start_size = 1;
size_t max_size = 1<<30;
// One rank will receive from all the others.
const int num_peers = 4;

template <typename Backend>
void do_recv(const typename VectorType<Backend>::type& expected_recv,
             int src,
             typename Backend::comm_type& comm) {
  auto recv = get_vector<Backend>(expected_recv.size());
  typename Backend::req_type req;
  Al::NonblockingSendRecv<Backend, float>(
    nullptr, 0, src,
    recv.data(), recv.size(), src,
    comm, req);
  Al::Wait<Backend>(req);
  if (!check_vector(expected_recv, recv)) {
    std::cout << comm.rank() << ": recv does not match" << std::endl;
    std::abort();
  }
}

template <typename Backend>
void do_send(typename VectorType<Backend>::type& to_send,
             const std::vector<int> dests,
             typename Backend::comm_type& comm) {
  std::vector<typename Backend::req_type> requests;
  const size_t chunk_per_peer = to_send.size() / dests.size();
  for (size_t i = 0; i < dests.size(); ++i) {
    int dest = dests[i];
    requests.push_back(Backend::null_req);
    auto& req = requests.back();
    Al::NonblockingSendRecv<Backend, float>(
      to_send.data() + i*chunk_per_peer, chunk_per_peer, dest,
      nullptr, 0, dest,
      comm, req);
  }
  for (auto& req : requests) {
    Al::Wait<Backend>(req);
  }
}

template <typename Backend>
void test_correctness() {
  typename Backend::comm_type comm = get_comm_with_stream<Backend>(MPI_COMM_WORLD);
  if (comm.size() % num_peers != 0) {
    std::cout << "Communicator size " << comm.size()
              << " not evenly divisible by " << num_peers
              << " peers" << std::endl;
    std::abort();
  }
  std::vector<size_t> sizes = get_sizes(start_size, max_size, true);
  if (comm.rank() % num_peers == 0) {
    // This rank sends everything.
    std::vector<int> peers;
    for (int peer = comm.rank() + 1; peer < comm.rank() + num_peers; ++peer) {
      peers.push_back(peer);
    }
    {
      std::stringstream ss;
      ss << comm.rank() << ": Peers: ";
      for (const auto& peer : peers) ss << peer << " ";
      std::cout << ss.str() << std::endl;
    }
    for (const auto& size : sizes) {
      if (comm.rank() == 0) {
        std::cout << "Testing size " << human_readable_size(size) << std::endl;
      }
      for (size_t trial = 0; trial < 1000; ++trial) {
        if (comm.rank() == 0) {
          std::cout << "Trial " << trial << std::endl;
        }
        const size_t total_size = size*peers.size();
        std::vector<float> host_to_send(total_size, comm.rank());
        typename VectorType<Backend>::type to_send(host_to_send);
        MPI_Barrier(MPI_COMM_WORLD);
        do_send<Backend>(to_send, peers, comm);
      }
    }
  } else {
    // Receive from sending rank.
    int peer = comm.rank() - (comm.rank() % num_peers);
    std::cout << comm.rank() << " Peers: " << peer << std::endl;
    for (const auto& size : sizes) {
      for (size_t trial = 0; trial < 1000; ++trial) {
        std::vector<float> host_to_recv(size, peer);
        typename VectorType<Backend>::type to_recv(host_to_recv);
        MPI_Barrier(MPI_COMM_WORLD);
        do_recv<Backend>(to_recv, peer, comm);
      }
    }
  }
  free_comm_with_stream<Backend>(comm);
}

int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
#ifdef AL_HAS_CUDA
  set_device();
#endif
  Al::Initialize(argc, argv);

  std::string backend = "MPI";
  parse_args(argc, argv, backend, start_size, max_size);

  if (backend == "MPI") {
    std::cerr << "Point-to-point not supported on MPI backend." << std::endl;
    std::abort();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    std::cerr << "Point-to-point not supported on NCCL backend." << std::endl;
    std::abort();
#endif
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    test_correctness<Al::MPICUDABackend>();
#endif
  }

  Al::Finalize();
  return 0;
}
