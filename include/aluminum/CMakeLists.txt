set_source_path(THIS_DIR_HEADERS
  Al.hpp
  internal.hpp
  progress.hpp
  base.hpp
  mempool.hpp
  mpi_impl.hpp
  profiling.hpp
  trace.hpp
  tuning_params.hpp
  utils.hpp
  )
set_source_path(THIS_DIR_CUDA_HEADERS
  cuda.hpp
  helper_kernels.hpp
  cudacommunicator.hpp
  )
set_source_path(THIS_DIR_MPI_CUDA_HEADERS
  mpi_cuda_impl.hpp
  )
set_source_path(THIS_DIR_HOST_TRANSFER_HEADERS
  ht_impl.hpp
  )
set_source_path(THIS_DIR_NCCL_HEADERS
  nccl_impl.hpp
  )
set_source_path(THIS_DIR_CUDA_AWARE_MPI_HEADERS
  cuda_aware_mpi_impl.hpp
  )

add_subdirectory(mpi)

if (AL_HAS_CUDA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_CUDA_HEADERS}")
endif ()

if (AL_HAS_MPI_CUDA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_MPI_CUDA_HEADERS}")

  add_subdirectory(mpi_cuda)
endif ()

if (AL_HAS_HOST_TRANSFER)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_HOST_TRANSFER_HEADERS}")
  add_subdirectory(ht)
endif ()

if (AL_HAS_NCCL)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_NCCL_HEADERS}")
endif ()

if (AL_HAS_CUDA_AWARE_MPI)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_CUDA_AWARE_MPI_HEADERS}")
  add_subdirectory(cuda_aware_mpi)
endif ()

set(ALUMINUM_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)

# Note (trb 07/27/2020): The way generated files work in CMake is a
# bit tricky -- they only work properly if they're used in the same
# scope that they are created. So I'm moving all the install logic to
# "/src/CMakeLists.txt", including the header installation. This
# doesn't really matter for CUDA, but it's a problem for HIP/ROCm.
