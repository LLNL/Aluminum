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

set(ALUMINUM_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)

if (AL_HAS_ROCM)
  hipify_header_files(ALUMINUM_HIP_HEADERS ${ALUMINUM_HEADERS})
  hipify_header_files(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
else ()
  set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
endif ()

install(FILES ${THIS_DIR_HEADERS_TO_INSTALL}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum)
