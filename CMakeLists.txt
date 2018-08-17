cmake_minimum_required(VERSION 3.9)
# 3.9 is required for CUDA as a first-class language.
#
# 3.8 is required for cxx_std_11 compile feature. Otherwise we could
# set the CMAKE_CXX_STANDARD flag, but the compile features are a more
# elegant solution.

project(ALUMINUM VERSION 0.1 LANGUAGES CXX)
# Not "CUDA" because no CUDA sources being compiled, so don't need
# NVCC, just cuda runtime.

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are prohibited. "
    "Create a new directory and build there.")
endif ()

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Options

option(ALUMINUM_ENABLE_CUDA "Enable CUDA support." OFF)
option(ALUMINUM_ENABLE_MPI_CUDA "Enable MPI-CUDA support." OFF)
option(ALUMINUM_ENABLE_NCCL "Enable NCCL support." OFF)
if (ALUMINUM_ENABLE_MPI_CUDA AND NOT ALUMINUM_ENABLE_CUDA)
  message(STATUS "MPI-CUDA support requested; enabling CUDA support, too.")
  set(ALUMINUM_ENABLE_CUDA ON)
endif ()
if (ALUMINUM_ENABLE_NCCL AND NOT ALUMINUM_ENABLE_CUDA)
  message(STATUS "NCCL support requested; enabling CUDA support, too.")
  set(ALUMINUM_ENABLE_CUDA ON)
endif ()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(AL_DEBUG ON)
endif ()

# Setup CXX requirements
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif ()

# Dependencies

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)

# Fix the imported target
get_property(_TMP_MPI_LINK_LIBRARIES TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES)
foreach(lib IN LISTS _TMP_MPI_LINK_LIBRARIES)
  if ("${lib}" MATCHES "-Wl*")
    list(APPEND _MPI_LINK_FLAGS "${lib}")
  else()
    list(APPEND _MPI_LINK_LIBRARIES "${lib}")
  endif ()
endforeach()

# The link flags *should* be propagated into this target somehow, but
# "LINK_FLAGS" is not a "whitelisted" property, so the INTERFACE
# target MPI::MPI_CXX cannot set them. But there's a clash with CUDA
# if they're added as "libraries".

#set_property(TARGET MPI::MPI_CXX PROPERTY LINK_FLAGS ${_MPI_LINK_FLAGS})
set_property(TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES ${_MPI_LINK_LIBRARIES})

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(HWLOC REQUIRED)

# Fix an issue with OpenMP's egregious use of non-language-specific
# compiler flags.
if (OPENMP_FOUND)
  get_target_property(
    __omp_compile_options OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:${__omp_compile_options}>)
  unset(__omp_compile_options)

  if (${CMAKE_CXX_COMPILER} MATCHES ".*xl[cC]")
    set_property(TARGET OpenMP::OpenMP_CXX APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES "${OpenMP_CXX_FLAGS}")
  endif ()
endif (OPENMP_FOUND)
if (ALUMINUM_ENABLE_CUDA)
  find_package(CUDA)
  if (CUDA_FOUND)
    enable_language(CUDA)
    set(AL_HAS_CUDA TRUE)
    if (ALUMINUM_ENABLE_MPI_CUDA)
      set(AL_HAS_CUDA TRUE)
      set(AL_HAS_MPI_CUDA TRUE)
    endif ()

    # Add the cuda imported target
    if (NOT TARGET cuda::cuda)
      add_library(cuda::cuda INTERFACE IMPORTED)
    endif ()

    set_property(TARGET cuda::cuda
      PROPERTY INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES})

    set_property(TARGET cuda::cuda
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS})

    if (ALUMINUM_ENABLE_NCCL)
      find_package(NCCL 2.0.0)
      if (NCCL_FOUND)
        set(AL_HAS_NCCL TRUE)
        set_property(TARGET cuda::cuda APPEND
          PROPERTY INTERFACE_LINK_LIBRARIES cuda::nccl)
      else ()
        message(WARNING "NCCL support requested but not found. Disabling.")
        set(ALUMINUM_ENABLE_NCCL OFF)
      endif ()
    endif ()
  else ()
    message(WARNING "CUDA support requested but not found. Disabling.")
    set(ALUMINUM_ENABLE_CUDA OFF)
    set(ALUMINUM_ENABLE_MPI_CUDA OFF)
    set(AL_HAS_CUDA FALSE)
    set(AL_HAS_MPI_CUDA FALSE)
  endif ()
endif ()
# Build library and executables

# Write the configuration file
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/Al_config.hpp.in"
  "${CMAKE_BINARY_DIR}/Al_config.hpp" @ONLY)

# Macro for setting up full paths
macro(set_full_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    unset(__name)
    get_filename_component(__name "${filename}" ABSOLUTE)
    list(APPEND __tmp_names "${__name}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

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
install(FILES
  "${CMAKE_BINARY_DIR}/Al_config.hpp" DESTINATION ${INCLUDE_INSTALL_DIRS})

# Install the CMake modules we need
install(FILES cmake/FindHWLOC.cmake cmake/FindNCCL.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})
