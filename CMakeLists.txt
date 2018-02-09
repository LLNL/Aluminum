cmake_minimum_required(VERSION 3.8)
# 3.8 is required for cxx_std_11 compile feature. Otherwise we could
# set the CMAKE_CXX_STANDARD flag, but the compile features are a more
# elegant solution.

project(ALUMINUM VERSION 0.1 LANGUAGES CXX)
# Not "CUDA" because no CUDA sources being compiled, so don't need
# NVCC, just cuda runtime.

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Options

option(ALUMINUM_ENABLE_CUDA "Enable CUDA support." OFF)
option(ALUMINUM_ENABLE_NCCL "Enable NCCL support." OFF)
if (ALUMINUM_ENABLE_NCCL AND NOT ALUMINUM_ENABLE_CUDA)
  message(status "NCCL support requested; enabling CUDA support, too.")
  set(ALUMINUM_ENABLE_CUDA ON)
endif ()

# Setup CXX requirements
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif ()

# Dependencies

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(HWLOC REQUIRED)

if (ALUMINUM_ENABLE_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    set(ALUMINUM_HAS_CUDA TRUE)
    if (NOT TARGET cuda::cuda)
      add_library(cuda::cuda INTERFACE IMPORTED)
    endif ()

    set_property(TARGET cuda::cuda
      PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES})

    set_property(TARGET cuda::cuda
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

    set_property(TARGET cuda::cuda
      PROPERTY INTERFACE_COMPILE_DEFINITIONS "ALUMINUM_HAS_CUDA")

    if (ALUMINUM_ENABLE_NCCL)
      find_package(NCCL 2.0.0)
      if (NCCL_FOUND)
        set(ALUMINUM_HAS_NCCL TRUE)
        set_property(TARGET cuda::cuda APPEND
          PROPERTY INTERFACE_LINK_LIBRARIES cuda::nccl)
        set_property(TARGET cuda::cuda APPEND
          PROPERTY INTERFACE_COMPILE_DEFINITIONS "ALUMINUM_HAS_NCCL")
      else ()
        message(WARNING "NCCL support requested but not found. Disabling.")
        set(ALUMINUM_ENABLE_NCCL OFF)
      endif ()
    endif ()
  else ()
    message(WARNING "CUDA support requested but not found. Disabling.")
    set(ALUMINUM_ENABLE_CUDA OFF)
  endif ()
endif ()

# Build library and executables

add_subdirectory(src)
add_subdirectory(benchmark)

# Testing
include(CTest)
add_subdirectory(test)

#
# Install target
#

include(CMakePackageConfigHelpers)

# Build directory
export(EXPORT AluminumTargets NAMESPACE AL:: FILE AluminumTargets.cmake)
write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/AluminumConfigVersion.cmake" VERSION
  ${ALUMINUM_VERSION} COMPATIBILITY AnyNewerVersion )

set(INCLUDE_INSTALL_DIRS ${CMAKE_SOURCE_DIR}/src)
set(LIB_INSTALL_DIR src)
set(CMAKE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_LOCATION "${CMAKE_SOURCE_DIR}/cmake")
set(REAL_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(cmake/AluminumConfig.cmake.in
  "${CMAKE_BINARY_DIR}/AluminumConfig.cmake" INSTALL_DESTINATION
  "${CMAKE_INSTALL_DIR}" PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)
set(CMAKE_INSTALL_PREFIX "${REAL_CMAKE_INSTALL_PREFIX}")

# Install directory

set(INCLUDE_INSTALL_DIRS ${CMAKE_INSTALL_INCLUDEDIR}/aluminum)
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/aluminum)
set(CMAKE_MODULE_LOCATION "\$\{CMAKE_CURRENT_LIST_DIR\}")
configure_package_config_file(cmake/AluminumConfig.cmake.in
  "${CMAKE_BINARY_DIR}/AluminumConfig.cmake.install" INSTALL_DESTINATION
  ${CMAKE_INSTALL_DIR} PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Install the install-tree files
install(FILES "${CMAKE_BINARY_DIR}/AluminumConfig.cmake.install"
  RENAME "AluminumConfig.cmake" DESTINATION ${CMAKE_INSTALL_DIR})
install(FILES
  "${CMAKE_BINARY_DIR}/AluminumConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})

# Install the CMake modules we need
install(FILES cmake/FindHWLOC.cmake cmake/FindNCCL.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})
