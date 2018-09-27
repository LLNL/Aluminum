set_full_path(THIS_DIR_HEADERS
  Al.hpp
  internal.hpp
  progress.hpp
  base.hpp
  mempool.hpp
  mpi_impl.hpp
  tuning_params.hpp
  )
set_full_path(THIS_DIR_CXX_SOURCES
  Al.cpp
  mpi_impl.cpp
  progress.cpp
  )

if (AL_HAS_CUDA)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/helper_kernels.hpp")
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/cudacommunicator.hpp")
  list(APPEND THIS_DIR_CXX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp")
  list(APPEND THIS_DIR_CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/helper_kernels.cu")
endif ()

if (AL_HAS_MPI_CUDA)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/mpi_cuda_impl.hpp")
  list(APPEND THIS_DIR_CXX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/mpi_cuda_impl.cpp")

  add_subdirectory(mpi_cuda)
endif ()

if (AL_HAS_NCCL)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/nccl_impl.hpp")
  list(APPEND THIS_DIR_CXX_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/nccl_impl.cpp")
endif ()

set(ALUMINUM_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}")
set(ALUMINUM_CXX_SOURCES "${CXX_SOURCES}" "${THIS_DIR_CXX_SOURCES}")
set(ALUMINUM_CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}")

add_library(Al
  "${ALUMINUM_CXX_SOURCES}" "${ALUMINUM_CUDA_SOURCES}" "${ALUMINUM_HEADERS}")
target_include_directories(Al PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/aluminum>)
target_link_libraries(Al PUBLIC
  MPI::MPI_CXX HWLOC::hwloc OpenMP::OpenMP_CXX
  $<$<BOOL:${AL_HAS_CUDA}>:cuda::cuda>)
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

install(FILES ${THIS_DIR_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum)
