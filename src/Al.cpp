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

#include <signal.h>
#include <execinfo.h>
#include <unistd.h>
#include <limits.h>
#include <string>
#include <sstream>
#include <fstream>
#include "Al.hpp"
#include "aluminum/internal.hpp"
#include "aluminum/progress.hpp"
#ifdef AL_HAS_CUDA
#include "aluminum/cuda.hpp"
#endif
#include "aluminum/trace.hpp"

namespace Al {

namespace {
// Whether the library has been initialized.
bool is_initialized = false;
// Progress engine.
internal::ProgressEngine* progress_engine = nullptr;

// Custom signal handler for debugging.
void handle_signal(int signal) {
  // Technically, these are unsafe to call from a signal handler.
  // But we burn everything at the end anyway, so this is best-effort.
  std::stringstream ss;
  ss << "Signal " << signal << " - ";
  switch (signal) {
  case SIGILL:
    ss << "illegal instruction";
    break;
  case SIGABRT:
    ss << "abort";
    break;
  case SIGFPE:
    ss << "floating point exception";
    break;
  case SIGBUS:
    ss << "bus error";
    break;
  case SIGSEGV:
    ss << "segmentation fault";
    break;
  default:
    ss << "unknown";
  }
  // Attempt a backtrace.
  ss << "\nBacktrace:\n";
  constexpr int max_frames = 128;
  void* frames[max_frames];
  int num_frames = backtrace(frames, max_frames);
  char** symbols = backtrace_symbols(frames, num_frames);
  for (int i = 0; i < num_frames; ++i) {
    ss << "\t" << i << ": ";
    if (symbols && symbols[i]) {
      ss << symbols[i];
    } else {
      ss << "(no symbol info)";
    }
    ss << "\n";
  }
  free(symbols);
  // Attempt to get progress engine state.
  if (progress_engine) {
    progress_engine->dump_state(ss);
  }
  // Write to a file.
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  pid_t pid = getpid();
  std::ofstream file(std::string(hostname) + "." + std::to_string(pid)
                     + ".dump.txt");
  file << ss.str();
  file.close();
#ifdef AL_TRACE
  internal::trace::write_trace_to_file();
#endif
}

}

void Initialize(int& argc, char**& argv) {
  // Avoid repeated initialization.
  if (is_initialized) {
    return;
  }
  internal::mpi::init(argc, argv);
  progress_engine = new internal::ProgressEngine();
  progress_engine->run();
  is_initialized = true;
#ifdef AL_HAS_CUDA
  internal::cuda::init(argc, argv);
#endif
#ifdef AL_HAS_NCCL
  internal::nccl::init(argc, argv);
#endif
#ifdef AL_HAS_MPI_CUDA
  internal::mpi_cuda::init(argc, argv);
#endif
#ifdef AL_HAS_HOST_TRANSFER
  internal::ht::init(argc, argv);
#endif

  // Add signal handlers.
  const std::vector<int> handled_signals = {SIGILL, SIGABRT, SIGFPE,
                                            SIGBUS, SIGSEGV};
  static struct sigaction sa;
  sa.sa_handler = &handle_signal;
  sa.sa_flags = SA_RESTART;
  sigfillset(&sa.sa_mask);
  for (const auto& sig : handled_signals) {
    sigaction(sig, &sa, nullptr);
  }
}

void Finalize() {
  // Make calling Finalize multiple times safely.
  if (!is_initialized) {
    return;
  }
  // Finalize in reverse order of initialization.
#ifdef AL_HAS_HOST_TRANSFER
  internal::ht::finalize();
#endif
#ifdef AL_HAS_MPI_CUDA
  internal::mpi_cuda::finalize();
#endif
#ifdef AL_HAS_NCCL
  internal::nccl::finalize();
#endif
#ifdef AL_HAS_CUDA
  internal::cuda::finalize();
#endif
  progress_engine->stop();
  delete progress_engine;
  progress_engine = nullptr;
  is_initialized = false;
  internal::mpi::finalize();
  internal::trace::write_trace_to_file();
}

bool Initialized() {
  return is_initialized;
}

namespace internal {

// Note: This is declared in progress.hpp.
ProgressEngine* get_progress_engine() {
  return progress_engine;
}

}  // namespace internal
}  // namespace Al
