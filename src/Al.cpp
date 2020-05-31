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
#include "internal.hpp"
#include "progress.hpp"
#ifdef AL_HAS_CUDA
#include "cuda.hpp"
#endif
#include "trace.hpp"

namespace Al {

namespace {
// Whether the library has been initialized.
bool is_initialized = false;
// Progress engine.
internal::ProgressEngine* progress_engine = nullptr;
// Whether this thread is in an error handler.
// Note this might not interoprate with anything not using real threads.
thread_local bool in_error_handler = false;

// Attempt to print a backtrace to os.
std::ostream& attempt_backtrace(std::ostream& os) {
  constexpr int max_frames = 128;
  void* frames[max_frames];
  int num_frames = backtrace(frames, max_frames);
  char** symbols = backtrace_symbols(frames, num_frames);
  os << "Backtrace\n";
  for (int i = 0; i < num_frames; ++i) {
    os << "\t" << i << ": ";
    if (symbols && symbols[i]) {
      os << symbols[i];
    } else {
      os << "(no symbol info)";
    }
    os << "\n";
  }
  free(symbols);
  return os;
}

// Write an error to a file.
void write_error_to_file(std::stringstream& ss) {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  pid_t pid = getpid();
  std::ofstream file(std::string(hostname) + "." + std::to_string(pid)
                     + ".dump.txt");
  file << ss.str();
  file.close();
}

// Dump an error.
void dump_error(std::stringstream& ss) {
  attempt_backtrace(ss);
  // Attempt to get progress engine state.
  if (progress_engine) {
    progress_engine->dump_state(ss);
  }
  write_error_to_file(ss);
#ifdef AL_TRACE
  internal::trace::write_trace_to_file();
#endif
}

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
  ss << "\n";
  dump_error(ss);
}

// Custom MPI error handler to print a backtrace.
void handle_mpi_error(MPI_Comm* comm, int* error, ...) {
  std::stringstream ss;
  // Catch recursion in case something goes wrong.
  if (in_error_handler) {
    ss << "MPI error " << *error << " while handling an error\n"
       << "Everything is probably on fire. Goodbye.";
    std::cerr << ss.str() << std::endl;
    std::abort();
  }

  in_error_handler = true;
  // Get the error string.
  char error_str[MPI_MAX_ERROR_STRING];
  int error_len;
  MPI_Error_string(*error, error_str, &error_len);
  // Get info about the communicator the error occurred on.
  int rank, size;
  MPI_Comm_rank(*comm, &rank);
  MPI_Comm_size(*comm, &size);
  ss << "MPI error " << *error
     << " on rank " << rank << " of " << size
     << ": " << error_str << "\n";
  // Print a note about the error.
  std::cerr << ss.str() << std::endl;
  dump_error(ss);
  std::exit(1);
}

}

void Initialize(int& argc, char**& argv) {
  // Avoid repeated initialization.
  if (is_initialized) {
    return;
  }
  internal::mpi::init(argc, argv);
  // Add error handler here, since we always use MPI and can use the error
  // helpers above.
  // This will be inherited by all MPI communicators.
  MPI_Errhandler errhandler;
  MPI_Comm_create_errhandler(&handle_mpi_error, &errhandler);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, errhandler);
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
