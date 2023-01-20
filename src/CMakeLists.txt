set_source_path(THIS_DIR_CXX_SOURCES
  Al.cpp
  mempool.cpp
  mpi_impl.cpp
  profiling.cpp
  progress.cpp
  trace.cpp
  )
set_source_path(THIS_DIR_MPI_CUDA_CXX_SOURCES
  mpi_cuda_impl.cpp
  )
set_source_path(THIS_DIR_HOST_TRANSFER_CXX_SOURCES
  ht_impl.cpp
  )
set_source_path(THIS_DIR_NCCL_CXX_SOURCES
  nccl_impl.cpp
  )

if (AL_HAS_CUDA)
  add_subdirectory(cuda)
endif ()

if (AL_HAS_MPI_CUDA)
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_MPI_CUDA_CXX_SOURCES}")

  add_subdirectory(mpi_cuda)
endif ()

if (AL_HAS_HOST_TRANSFER)
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_HOST_TRANSFER_CXX_SOURCES}")
endif ()

if (AL_HAS_NCCL)
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_NCCL_CXX_SOURCES}")
endif ()

set(ALUMINUM_CXX_SOURCES "${CXX_SOURCES}" "${THIS_DIR_CXX_SOURCES}")
set(ALUMINUM_CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}")

add_library(Al
  "${ALUMINUM_CXX_SOURCES}"
  "${ALUMINUM_CUDA_SOURCES}"
  "${ALUMINUM_HEADERS}")
target_include_directories(Al PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

set_target_properties(Al
  PROPERTIES
  VERSION ${ALUMINUM_VERSION})

target_compile_features(Al PUBLIC cxx_std_17)
target_include_directories(Al PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(Al PUBLIC
  HWLOC::hwloc
  MPI::MPI_CXX
  Threads::Threads)

if (AL_HAS_ROCM)
  set(HIP_CLANG_ROOT "${AL_ROCM_PATH}/llvm")

  ##############################
  # FIXME: This bit is to fix errors in the HIP CMake config
  #        file. Hopefully this can be removed in a future version of
  #        ROCm.
  file(GLOB HIP_CLANG_INCLUDE_SEARCH_PATHS
    "${HIP_CLANG_ROOT}/lib/clang/*/include")
  find_path(HIP_CLANG_INCLUDE_PATH stddef.h
    HINTS "${HIP_CLANG_INCLUDE_SEARCH_PATHS}"
    NO_DEFAULT_PATH)

  file(GLOB HIP_CLANGRT_LIB_SEARCH_PATHS
    "${HIP_CLANG_ROOT}/lib/clang/*/lib/*")
  find_library(ACTUAL_CLANGRT_BUILTINS clangrt-builtins
    NAMES
    clang_rt.builtins
    clang_rt.builtins-x86_64
    PATHS
    "${HIP_CLANGRT_LIB_SEARCH_PATHS}")

  get_target_property(_HIP_HOST_LIBS hip::host INTERFACE_LINK_LIBRARIES)
  get_target_property(_HIP_DEVICE_LIBS hip::device INTERFACE_LINK_LIBRARIES)

  string(REPLACE
    "CLANGRT_BUILTINS-NOTFOUND"
    "${ACTUAL_CLANGRT_BUILTINS}"
    _NEW_HIP_HOST_LIBS
    "${_HIP_HOST_LIBS}")
  string(REPLACE
    "CLANGRT_BUILTINS-NOTFOUND"
    "${ACTUAL_CLANGRT_BUILTINS}"
    _NEW_HIP_DEVICE_LIBS
    "${_HIP_DEVICE_LIBS}")

  set_property(TARGET hip::host
    PROPERTY INTERFACE_LINK_LIBRARIES ${_NEW_HIP_HOST_LIBS})
  set_property(TARGET hip::device
    PROPERTY INTERFACE_LINK_LIBRARIES ${_NEW_HIP_DEVICE_LIBS})
  # END FIXME BLOCK
  ##############################

  target_include_directories(Al PRIVATE
    $<BUILD_INTERFACE:${ROCM_SMI_INCLUDE_DIR}>)
  target_link_libraries(Al PUBLIC
    hip::host
    hip::hipcub
    $<$<BOOL:${AL_HAS_NCCL}>:roc::rccl>
    roc::rocprim
    ${ROCM_SMI_LIBRARY}
    ${Roctracer_LIBRARIES})

  # HIP language files are detected by the ".hip" extension. Since
  # we're upcycling ".cu" files for now, we just set their LANGUAGE to
  # be HIP.
  get_target_property(AL_HIP_SOURCES Al SOURCES)
  list(FILTER AL_HIP_SOURCES INCLUDE REGEX ".cu$")
  set_source_files_properties(${AL_HIP_SOURCES}
    PROPERTIES LANGUAGE HIP)

elseif (AL_HAS_CUDA)
  target_link_libraries(Al PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    $<$<BOOL:${AL_HAS_NVPROF}>:CUDA::nvToolsExt>
    $<$<BOOL:${AL_HAS_NCCL}>:cuda::nccl>
    $<$<BOOL:${AL_HAS_EXTERNAL_CUB}>:cuda::CUB>)
  target_compile_features(Al PUBLIC cuda_std_14)
endif (AL_HAS_ROCM)

# Header installation.
install(
  DIRECTORY ${CMAKE_SOURCE_DIR}/include/aluminum
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp")
install(
  FILES ${CMAKE_SOURCE_DIR}/include/Al.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS Al
  EXPORT AluminumTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(EXPORT AluminumTargets
  NAMESPACE AL::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aluminum
  )
