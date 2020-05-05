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
set_source_path(THIS_DIR_CXX_SOURCES
  Al.cpp
  mpi_impl.cpp
  profiling.cpp
  progress.cpp
  trace.cpp
  )
set_source_path(THIS_DIR_CUDA_HEADERS
  cuda.hpp
  helper_kernels.hpp
  cudacommunicator.hpp
  )
set_source_path(THIS_DIR_CUDA_CXX_SOURCES
  cuda.cpp
  )
set_source_path(THIS_DIR_CUDA_SOURCES
  helper_kernels.cu
  )
set_source_path(THIS_DIR_MPI_CUDA_HEADERS
  mpi_cuda_impl.hpp
  )
set_source_path(THIS_DIR_MPI_CUDA_CXX_SOURCES
  mpi_cuda_impl.cpp
  )
set_source_path(THIS_DIR_NCCL_HEADERS
  nccl_impl.hpp
  )
set_source_path(THIS_DIR_NCCL_CXX_SOURCES
  nccl_impl.cpp
  )

if (AL_HAS_CUDA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_CUDA_HEADERS}")
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_CUDA_CXX_SOURCES}")
  list(APPEND THIS_DIR_CUDA_SOURCES "${THIS_DIR_CUDA_SOURCES}")
endif ()

if (AL_HAS_MPI_CUDA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_MPI_CUDA_HEADERS}")
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_MPI_CUDA_CXX_SOURCES}")

  add_subdirectory(mpi_cuda)
endif ()

if (AL_HAS_NCCL)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_NCCL_HEADERS}")
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_NCCL_CXX_SOURCES}")
endif ()

set(ALUMINUM_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}")
set(ALUMINUM_CXX_SOURCES "${CXX_SOURCES}" "${THIS_DIR_CXX_SOURCES}")
set(ALUMINUM_CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}")

if (AL_HAS_ROCM)
  hipify_source_files(ALUMINUM_HIP_SOURCES ${ALUMINUM_CXX_SOURCES} ${ALUMINUM_CUDA_SOURCES})
  hipify_header_files(ALUMINUM_HIP_HEADERS ${ALUMINUM_HEADERS})

  hip_add_library(Al "${ALUMINUM_HIP_SOURCES}" "${ALUMINUM_HIP_HEADERS}")
  target_include_directories(Al PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

  hipify_header_files(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
else ()
  add_library(Al
    "${ALUMINUM_CXX_SOURCES}" "${ALUMINUM_CUDA_SOURCES}" "${ALUMINUM_HEADERS}")
  target_link_libraries(Al PUBLIC
    $<$<BOOL:${AL_HAS_CUDA}>:cuda::cuda>
    $<$<BOOL:${AL_HAS_NVPROF}>:cuda::nvtx>)
  target_include_directories(Al PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

  set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
endif ()

target_include_directories(Al PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/aluminum>)
target_link_libraries(Al PUBLIC
  MPI::MPI_CXX HWLOC::hwloc
  $<$<BOOL:${AL_HAS_OPENMP}>:OpenMP::OpenMP_CXX>
  $<TARGET_NAME_IF_EXISTS:roc::rccl>)

target_compile_features(Al PUBLIC cxx_std_11)

install(TARGETS Al
  EXPORT AluminumTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum
  )

install(EXPORT AluminumTargets
  NAMESPACE AL::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aluminum
  )

install(FILES ${THIS_DIR_HEADERS_TO_INSTALL}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum)
