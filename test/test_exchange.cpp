////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include "Al.hpp"
#include <iostream>
#include <cxxopts.hpp>
#include "test_utils.hpp"


template <typename Backend, typename T,
          std::enable_if_t<IsTypeSupported<Backend, T>::value, bool> = true>
void run_test(cxxopts::ParseResult& parsed_opts) {
  CommWrapper<Backend> comm_wrapper(MPI_COMM_WORLD);
  const int num_peers = parsed_opts["num-peers"].as<int>();
  if (comm_wrapper.comm().size() <= num_peers) {
    std::cerr << "Communicator size " << comm_wrapper.comm().size()
              << " too small for " << num_peers << " peers" << std::endl;
    std::abort();
  }

  const int peers_per_side = num_peers / 2;
  // Set up options for each peer.
  std::vector<OpOptions<Backend>> peers;
  for (int peer = comm_wrapper.comm().rank() - peers_per_side;
       peer <= comm_wrapper.comm().rank() + peers_per_side;
       ++peer) {
    if (peer < 0
        || peer >= comm_wrapper.comm().size()
        || peer == comm_wrapper.comm().rank()) {
      continue;
    }
    peers.emplace_back();
    peers.back().src = peer;
    peers.back().dst = peer;
    // The exchanges are always nonblocking.
    peers.back().nonblocking = true;
  }

  auto sizes = get_sizes_from_opts(parsed_opts);
  for (const auto& size : sizes) {
    // Each rank sends vectors filled with its rank as data.
    std::vector<typename VectorType<T, Backend>::type> inputs(peers.size());
    std::vector<typename VectorType<T, Backend>::type> outputs(peers.size());
    std::vector<std::vector<T>> expected_inputs(peers.size());
    std::vector<std::vector<T>> expected_outputs(peers.size());
    for (size_t i = 0; i < peers.size(); ++i) {
      expected_inputs[i] = std::vector<T>(size, comm_wrapper.comm().rank());
      expected_outputs[i] = std::vector<T>(size, peers[i].src);
      inputs[i] = VectorType<T, Backend>::type(expected_inputs[i]);
      outputs[i] = get_vector<T, Backend>(size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Start the exchanges.
    for (size_t i = 0; i < peers.size(); ++i) {
      OpDispatcher<Backend, T> op_runner(AlOperation::sendrecv, peers[i]);
      op_runner.run(inputs[i], outputs[i], comm_wrapper.comm());
    }
    // Wait for all to complete.
    for (size_t i = 0; i < peers.size(); ++i) {
      Al::Wait<Backend>(peers[i].req);
    }
    // Check all results.
    for (size_t i = 0; i < peers.size(); ++i) {
      if (!check_vector(expected_inputs[i], inputs[i])) {
        std::cerr << comm_wrapper.comm().rank()
                  << ": input does not match for size "
                  << size << std::endl;
        std::abort();
      }
      if (!check_vector(expected_outputs[i], outputs[i])) {
        std::cerr << comm_wrapper.comm().rank()
                  << ": output does not match for size "
                  << size << std::endl;
        std::abort();
      }
    }
  }
}

template <typename Backend, typename T,
          std::enable_if_t<!IsTypeSupported<Backend, T>::value, bool> = true>
void run_test(cxxopts::ParseResult& parsed_opts) {
  std::cerr << "Backend "
            << parsed_opts["backend"].as<std::string>()
            << " does not support datatype "
            << parsed_opts["datatype"].as<std::string>() << std::endl;
  std::abort();
}

struct test_dispatcher {
  template <typename Backend, typename T>
  void operator()(cxxopts::ParseResult& parsed_opts) {
    run_test<Backend, T>(parsed_opts);
  }
};

int main(int argc, char** argv) {
  test_init_aluminum(argc, argv);

  cxxopts::Options options("test_exchange", "Test halo exchange pattern");
  options.add_options()
    ("backend", "Aluminum backend", cxxopts::value<std::string>())
    ("num-peers", "Number of peers to exchange with", cxxopts::value<int>()->default_value("4"))
    ("size", "Size of message to test", cxxopts::value<size_t>())
    ("min-size", "Minimum size of message to test", cxxopts::value<size_t>()->default_value("1"))
    ("max-size", "Maximum size of message to test", cxxopts::value<size_t>()->default_value("4194304"))
    ("datatype", "Message datatype", cxxopts::value<std::string>()->default_value("float"))
    ("hang-rank", "Hang a specific or all ranks at startup", cxxopts::value<int>()->default_value("-1"));
  auto parsed_opts = options.parse(argc, argv);

  if (parsed_opts.count("hang-rank")) {
    hang_for_debugging(parsed_opts["hang-rank"].as<int>());
  }

  // Simple validation.
  if (!parsed_opts.count("backend")) {
    std::cerr << "Must provide a backend to use" << std::endl;
    test_fini_aluminum();
    return EXIT_FAILURE;
  }

  test_fini_aluminum();
  return 0;
}
