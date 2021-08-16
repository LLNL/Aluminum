cmake_minimum_required(VERSION 3.17)
# 3.9 is required for CUDA as a first-class language.
#
# 3.8 is required for cxx_std_11 compile feature. Otherwise we could
# set the CMAKE_CXX_STANDARD flag, but the compile features are a more
# elegant solution.

#
# Version setup
#

set(ALUMINUM_VERSION_MAJOR 1)
set(ALUMINUM_VERSION_MINOR 0)
set(ALUMINUM_VERSION_PATCH 0)
set(ALUMINUM_VERSION "${ALUMINUM_VERSION_MAJOR}.${ALUMINUM_VERSION_MINOR}.${ALUMINUM_VERSION_PATCH}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Configuration type" FORCE)
endif ()

project(ALUMINUM VERSION ${ALUMINUM_VERSION} LANGUAGES CXX)
# Not "CUDA" just yet since that's only one possible device paradigm.

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are prohibited. "
    "Create a new directory and build there.")
endif ()

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Options

option(ALUMINUM_ENABLE_CUDA "Enable CUDA support." OFF)
option(ALUMINUM_ENABLE_ROCM "Enable HIP/ROCm support." OFF)

# CUDA and ROCm are considered to be mutually exclusive.
if (ALUMINUM_ENABLE_CUDA AND ALUMINUM_ENABLE_ROCM)
  message(FATAL_ERROR
    "CUDA and HIP/ROCm support are mutually exclusive. "
    "Please only enable one.")
endif ()

# The first of many atrocities to come: the build will be largely
# agnostic to which of CUDA or ROCm is enabled. Use a generic flag to
# hide this.
if (ALUMINUM_ENABLE_CUDA OR ALUMINUM_ENABLE_ROCM)
  set(ALUMINUM_HAS_GPU ON)
endif ()

# If ROCm is enabled, this means "ALUMINUM_ENABLE_MPI_ROCM"...
option(ALUMINUM_ENABLE_MPI_CUDA "Enable MPI-CUDA support." OFF)
option(ALUMINUM_ENABLE_MPI_CUDA_RMA "Enable RMA in MPI-CUDA." OFF)
option(ALUMINUM_ENABLE_HOST_TRANSFER "Enable host-transfer support." OFF)

if (ALUMINUM_ENABLE_MPI_CUDA_RMA AND NOT ALUMINUM_ENABLE_MPI_CUDA)
  message(STATUS
    "RMA in MPI-CUDA requested; enabling MPI-CUDA support, too.")
  set(ALUMINUM_ENABLE_MPI_CUDA ON)
endif ()

if (ALUMINUM_ENABLE_MPI_CUDA AND NOT ALUMINUM_HAS_GPU)
  message(STATUS
    "MPI-CUDA support requested but no GPU runtime enabled. "
    "Assuming CUDA support.")
  set(ALUMINUM_ENABLE_CUDA ON)
endif ()

if (ALUMINUM_ENABLE_HOST_TRANSFER AND NOT ALUMINUM_HAS_GPU)
  message(STATUS
    "Host-transfer support requested but no GPU runtime enabled. "
    "Assuming CUDA support.")
  set(ALUMINUM_ENABLE_CUDA ON)
endif ()

# If ROCm is enabled, this means "ALUMINUM_ENABLE_RCCL"...
option(ALUMINUM_ENABLE_NCCL "Enable NCCL support." OFF)

if (ALUMINUM_ENABLE_NCCL AND NOT ALUMINUM_HAS_GPU)
  message(STATUS
    "NCCL support requested but no GPU runtime enabled. "
    "Assuming CUDA support.")
  set(ALUMINUM_ENABLE_CUDA ON)
endif ()

if (ALUMINUM_ENABLE_CUDA OR ALUMINUM_ENABLE_ROCM)
  set(ALUMINUM_HAS_GPU ON)
endif ()

option(ALUMINUM_DEBUG_HANG_CHECK
  "Enable hang checking."
  OFF)
option(ALUMINUM_ENABLE_NVPROF
  "Enable profiling via nvprof/NVTX."
  OFF)
option(ALUMINUM_ENABLE_STREAM_MEM_OPS
  "Enable stream memory operations."
  OFF)
option(ALUMINUM_ENABLE_THREAD_MULTIPLE
  "Allow multiple threads to call Aluminum concurrently."
  OFF)
option(ALUMINUM_ENABLE_TRACE
  "Enable runtime tracing."
  OFF)

option(ALUMINUM_MPI_SERIALIZE
  "Serialize MPI operations."
  OFF)

option(ALUMINUM_ENABLE_TESTS
  "Build tests."
  OFF)
option(ALUMINUM_ENABLE_BENCHMARKS
  "Build benchmarks."
  OFF)

if (ALUMINUM_HAS_GPU
    AND NOT ALUMINUM_ENABLE_NCCL
    AND NOT ALUMINUM_ENABLE_MPI_CUDA
    AND NOT ALUMINUM_ENABLE_HOST_TRANSFER)
  message(FATAL_ERROR
    "CUDA or ROCm has been enabled without a backend. "
    "This should not happen. "
    "Please turn on \"ALUMINUM_ENABLE_NCCL\" and/or "
    "\"ALUMINUM_ENABLE_MPI_CUDA\" and/or "
    "\"ALUMINUM_ENABLE_HOST_TRANSFER\" and reconfigure.")
endif ()

string(TOUPPER "${CMAKE_BUILD_TYPE}" AL_BUILD_TYPE_UPPER)
if (AL_BUILD_TYPE_UPPER MATCHES "DEBUG")
  set(AL_DEBUG ON)
  set(AL_ENABLE_TESTS ON)
endif ()
if (ALUMINUM_DEBUG_HANG_CHECK)
  set(AL_DEBUG_HANG_CHECK ON)
endif ()
if (ALUMINUM_ENABLE_STREAM_MEM_OPS)
  set(AL_USE_STREAM_MEM_OPS ON)
endif ()
if (ALUMINUM_ENABLE_THREAD_MULTIPLE)
  set(AL_THREAD_MULTIPLE ON)
endif ()
if (ALUMINUM_ENABLE_TRACE)
  set(AL_TRACE ON)
endif ()
if (ALUMINUM_MPI_SERIALIZE)
  set(AL_MPI_SERIALIZE ON)
endif ()
if (ALUMINUM_ENABLE_TESTS)
  set(AL_ENABLE_TESTS ON)
endif ()
if (ALUMINUM_ENABLE_BENCHMARKS)
  set(AL_ENABLE_BENCHMARKS ON)
endif ()

# Setup CXX requirements
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
# -faligned-new is needed to use new alignment-aware new when available.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-faligned-new" CXX_COMPILER_HAS_FALIGNED_NEW)
if (CXX_COMPILER_HAS_FALIGNED_NEW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
endif ()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Set -g3 if supported to get better debugging experience.
check_cxx_compiler_flag("-g3" CXX_COMPILER_HAS_G3)
if (CXX_COMPILER_HAS_G3)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")
else ()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif ()
# Set -Og for debug builds when supported.
check_cxx_compiler_flag("-Og" CXX_COMPILER_HAS_OG)
if (CXX_COMPILER_HAS_OG)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
endif ()

if (NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif ()

# Dependencies

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Fix the imported target

# FIXME (trb): We should split the library into language-specific
# targets. That is, the .cu files should never need MPI linkage, so
# they should be built into a separate target without MPI::MPI_CXX
# "linkage".
get_target_property(
  __mpi_compile_options MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
if (__mpi_compile_options)
  set_property(TARGET MPI::MPI_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:${__mpi_compile_options}>)
  unset(__mpi_compile_options)
endif ()

# Assuming this target is excluded from all device-link steps, this
# should not be necessary after the above FIXME is resolved.
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

if (ALUMINUM_ENABLE_ROCM)
  set(AL_HWLOC_MINIMUM_VERSION "2.3.0")
  find_package(HWLOC "${AL_HWLOC_MINIMUM_VERSION}" REQUIRED)
else ()
  find_package(HWLOC REQUIRED)
endif ()

if (ALUMINUM_ENABLE_CUDA)
  find_package(CUDA 9.0)
  if (CUDA_FOUND)
    enable_language(CUDA)
    set(AL_HAS_CUDA TRUE)

    if (ALUMINUM_ENABLE_MPI_CUDA)
      set(AL_HAS_CUDA TRUE)
      set(AL_HAS_MPI_CUDA TRUE)
      if (ALUMINUM_ENABLE_MPI_CUDA_RMA)
        set(AL_HAS_MPI_CUDA_RMA TRUE)
      endif ()
    endif ()

    if (ALUMINUM_ENABLE_HOST_TRANSFER)
      set(AL_HAS_CUDA TRUE)
      set(AL_HAS_HOST_TRANSFER TRUE)
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
      find_package(NCCL 2.7.0 REQUIRED)

      set(AL_HAS_NCCL TRUE)
      set_property(TARGET cuda::cuda APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES cuda::nccl)
    endif ()

    if (ALUMINUM_ENABLE_NVPROF)
      find_package(NVTX REQUIRED)
      set(AL_HAS_NVPROF ON)
    endif ()

    find_package(CUB REQUIRED)
    set_property(TARGET cuda::cuda APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES cuda::CUB)
  else ()
    message(WARNING "CUDA support requested but not found. Disabling.")
    set(ALUMINUM_ENABLE_CUDA OFF)
    set(ALUMINUM_ENABLE_MPI_CUDA OFF)
    set(ALUMINUM_ENABLE_HOST_TRANSFER OFF)
    set(ALUMINUM_ENABLE_NCCL OFF)
    set(ALUMINUM_ENABLE_NVPROF OFF)
    set(AL_HAS_CUDA FALSE)
    set(AL_HAS_MPI_CUDA FALSE)
    set(AL_HAS_HOST_TRANSFER FALSE)
    set(AL_HAS_NCCL FALSE)
    set(AL_HAS_NVPROF OFF)
  endif ()

  # Check that a backend is found
  if (NOT AL_HAS_NCCL AND NOT AL_HAS_MPI_CUDA AND NOT AL_HAS_HOST_TRANSFER)
    set(ALUMINUM_ENABLE_CUDA FALSE)
    set(AL_HAS_CUDA FALSE)
  endif ()
endif ()

if (ALUMINUM_ENABLE_ROCM)

  list(APPEND AL_HIP_HIPCC_FLAGS "-std=c++17")
  if (CMAKE_POSITION_INDEPENDENT_CODE)
    list(APPEND AL_HIP_HIPCC_FLAGS "-fPIC")
  endif ()
  set(HIP_HIPCC_FLAGS "${AL_HIP_HIPCC_FLAGS};${HIP_HIPCC_FLAGS}"
    CACHE STRING "Semi-colon delimited list of flags to pass to hipcc"
    FORCE) # <- because these are definitely the flags we want.

  if (ROCM_PATH)
    set(AL_ROCM_PATH "${ROCM_PATH}")
  elseif (DEFINED ENV{ROCM_PATH})
    set(AL_ROCM_PATH "$ENV{ROCM_PATH}")
  else ()
    # We could add a fallback here trying to find the base of some
    # likely-to-exist executable like "hipconfig" or "hipcc" or even
    # "rocm-smi" or something. Or the user can just use the ROCM_PATH
    # options above.
    set(AL_ROCM_PATH "/opt/rocm")
  endif ()
  message(STATUS "Using AL_ROCM_PATH: ${AL_ROCM_PATH}")

  # Provides hipify_*_files
  include(HipBuildSystem)

  # Provides hip_add_executable
  set(CMAKE_MODULE_PATH "${AL_ROCM_PATH}/hip/cmake" ${CMAKE_MODULE_PATH})

  # hip-runtime library
  find_package(HIP MODULE REQUIRED)
  set(AL_HAS_ROCM TRUE)

  # This is needed for some compatibility
  # things in the source code.
  set(AL_HAS_CUDA TRUE)

  if (ALUMINUM_ENABLE_MPI_CUDA)
    set(AL_HAS_MPI_CUDA TRUE)
    if (ALUMINUM_ENABLE_MPI_CUDA_RMA)
      set(AL_HAS_MPI_CUDA_RMA TRUE)
    endif ()
  endif ()

  if (ALUMINUM_ENABLE_HOST_TRANSFER)
    set(AL_HAS_HOST_TRANSFER TRUE)
  endif ()

  if (ALUMINUM_ENABLE_NCCL)
    find_package(rccl CONFIG QUIET
      HINTS ${RCCL_DIR} $ENV{RCCL_DIR}
      PATH_SUFFIXES lib64/cmake/rccl lib/cmake/rccl
      NO_DEFAULT_PATH)
    find_package(rccl CONFIG REQUIRED)
    message(STATUS "Found RCCL: ${rccl_DIR}")
    set(AL_HAS_NCCL TRUE)
  endif ()

  # Find CUB
  set(CMAKE_PREFIX_PATH "${AL_ROCM_PATH}/hip" ${CMAKE_PREFIX_PATH})
  set(HIP_FOUND FALSE)
  find_package(HIP CONFIG REQUIRED)
  find_package(rocPRIM REQUIRED)
  find_package(hipCUB REQUIRED)

  # (trb 11/02/2020): This is not ideal; when things stabilize, I'll
  # clean this up. Ideally we'll have a CMake config file for this.
  find_path(ROCM_SMI_INCLUDE_DIR rocm_smi/rocm_smi.h REQUIRED
    HINTS ${AL_ROCM_PATH}/rocm_smi/include)

  find_library(ROCM_SMI_LIBRARY rocm_smi64 REQUIRED
    HINTS ${AL_ROCM_PATH}/rocm_smi/lib)

  # Just in case any of the previous commands turned this on.
  set(CMAKE_CXX_EXTENSIONS FALSE)

  # For some reason, hipified files can be relatively specified, but regular
  # files must be given a full path. This appears to be the easiest way for both
  # ROCm and non-ROCm builds to both work and install.
  set(SOURCE_PREFIX "")
else ()
  set(SOURCE_PREFIX "${CMAKE_CURRENT_LIST_DIR}/")
endif ()

# Pull the Git submodules.
# Adapted from: https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if (NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
  endif ()
endif ()

# Build library and executables

# Write the configuration file
# Update to include other profiling interfaces as needed:
if (AL_HAS_NVPROF)
  set(AL_HAS_PROF ON)
endif ()
configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/Al_config.hpp.in"
  "${CMAKE_BINARY_DIR}/Al_config.hpp" @ONLY)

# Macro for setting up paths
macro(set_source_path VAR)
  unset(__tmp_names)

  file(RELATIVE_PATH __relative_to
    "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_LIST_DIR}")

  foreach(filename ${ARGN})
    unset(__name)
    get_filename_component(__name "${filename}" NAME)
    list(APPEND __tmp_names "${SOURCE_PREFIX}${__relative_to}/${__name}")
    #message(DEBUG "Set source path of ${__name} to ${SOURCE_PREFIX}${__relative_to}/${__name}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

add_subdirectory(include/aluminum)

# Add in the master header
list(APPEND ALUMINUM_HEADERS "${CMAKE_SOURCE_DIR}/include/Al.hpp")

# Add the library targets
add_subdirectory(src)

# Testing
include(CTest)
add_subdirectory(test)

# The benchmarks depend on some test utility headers, so it must come
# after the test/ directory.
if (AL_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
endif ()

#
# Install target
#

include(CMakePackageConfigHelpers)

# Build directory
export(EXPORT AluminumTargets NAMESPACE AL:: FILE AluminumTargets.cmake)
write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/AluminumConfigVersion.cmake" VERSION
  ${ALUMINUM_VERSION} COMPATIBILITY SameMinorVersion )

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

set(INCLUDE_INSTALL_DIRS ${CMAKE_INSTALL_INCLUDEDIR})
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
install(FILES
  cmake/FindHWLOC.cmake
  cmake/FindNCCL.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})
