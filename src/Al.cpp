#include "Al.hpp"
#include "internal.hpp"
#include "progress.hpp"
#ifdef AL_HAS_CUDA
#include "cuda.hpp"
#endif

namespace Al {

namespace {
// Whether the library has been initialized.
bool is_initialized = false;
// Progress engine.
internal::ProgressEngine* progress_engine = nullptr;
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
