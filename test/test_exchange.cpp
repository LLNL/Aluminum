#include <iostream>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_ht.hpp"
#endif

#include <stdlib.h>
#include <math.h>
#include <string>
#include <sstream>

size_t start_size = 1;
size_t max_size = 1<<30;
const int num_peers = 4;  // Should be even, but will not wrap ranks.

template <typename Backend>
void do_transfer(const typename VectorType<Backend>::type& expected_recv,
                 typename VectorType<Backend>::type& to_send,
                 const std::vector<int> peers,
                 typename Backend::comm_type& comm) {
  auto recv = get_vector<Backend>(expected_recv.size());
  std::vector<typename Backend::req_type> requests;
  const size_t chunk_per_peer = to_send.size() / peers.size();
  for (size_t i = 0; i < peers.size(); ++i) {
    int peer = peers[i];
    requests.push_back(Backend::null_req);
    auto& req = requests.back();
    Al::NonblockingSendRecv<Backend>(
      to_send.data() + i*chunk_per_peer, chunk_per_peer, peer,
      recv.data() + i*chunk_per_peer, chunk_per_peer, peer,
      comm, req);
  }
  for (auto& req : requests) {
    Al::Wait<Backend>(req);
  }
  if (!check_vector(expected_recv, recv)) {
    std::cout << comm.rank() << ": sendrecv does not match" << std::endl;
    std::abort();
  }
}

template <typename Backend>
void test_correctness() {
  typename Backend::comm_type comm = get_comm_with_stream<Backend>(MPI_COMM_WORLD);
  if (comm.size() <= num_peers) {
    std::cout << "Communicator size " << comm.size() << " too small for "
              << num_peers << " peers" << std::endl;
    std::abort();
  }
  const int peers_per_side = num_peers / 2;
  std::vector<int> peers;
  for (int peer = comm.rank() - peers_per_side;
       peer <= comm.rank() + peers_per_side;
       ++peer) {
    if (peer < 0 || peer >= comm.size() || peer == comm.rank()) {
      continue;
    }
    peers.push_back(peer);
  }
  {
    std::stringstream ss;
    ss << comm.rank() << ": Peers: ";
    for (const auto& peer : peers) ss << peer << " ";
    std::cout << ss.str() << std::endl;
  }
  // Compute sizes to test.
  std::vector<size_t> sizes = get_sizes(start_size, max_size, true);
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    for (size_t trial = 0; trial < 1; ++trial) {
      const size_t total_size = size*peers.size();
      std::vector<float> host_to_send(total_size, comm.rank());
      std::vector<float> host_to_recv(total_size);
      // Compute correct values.
      for (size_t i = 0; i < peers.size(); ++i) {
        int peer = peers[i];
        std::fill_n(host_to_recv.data() + i*size, size, peer);
      }
      typename VectorType<Backend>::type to_send(host_to_send);
      typename VectorType<Backend>::type to_recv(host_to_recv);
      MPI_Barrier(MPI_COMM_WORLD);
      do_transfer<Backend>(to_recv, to_send, peers, comm);
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
    test_correctness<Al::MPIBackend>();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    test_correctness<Al::NCCLBackend>();
#endif
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    std::cerr << "Point-to-point not supported on MPI-CUDA backend." << std::endl;
    std::abort();
#endif
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    test_correctness<Al::HostTransferBackend>();
#endif
  }

  Al::Finalize();
  return 0;
}
