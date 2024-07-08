cmake_minimum_required(VERSION 3.21)
# 3.21 fixes an important bug related to FindCUDAToolkit and the
# NVIDIA HPC SDK (Perlmutter, e.g.)

# Version setup
set(ALUMINUM_VERSION_MAJOR 1)
set(ALUMINUM_VERSION_MINOR 4)
set(ALUMINUM_VERSION_PATCH 2)
set(ALUMINUM_VERSION
  "${ALUMINUM_VERSION_MAJOR}.${ALUMINUM_VERSION_MINOR}.${ALUMINUM_VERSION_PATCH}")

# Gather Git information if available.
find_package(Git)
if (GIT_EXECUTABLE)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --is-inside-source-tree
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    OUTPUT_VARIABLE __BUILDING_FROM_GIT_SOURCES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (__BUILDING_FROM_GIT_SOURCES)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
      WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
      OUTPUT_VARIABLE __GIT_TOPLEVEL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --git-dir
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_GIT_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} --git-dir "${__GIT_GIT_DIR}" describe
      --abbrev=7 --always --dirty --tags
      WORKING_DIRECTORY "${__GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE __GIT_DESCRIBE_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(ALUMINUM_GIT_VERSION "${__GIT_DESCRIBE_VERSION}")
  endif ()
endif ()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Configuration type" FORCE)
endif ()

project(ALUMINUM VERSION ${ALUMINUM_VERSION} LANGUAGES CXX)
# Not "CUDA" just yet since that's only one possible device paradigm.

if (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are prohibited. "
    "Create a new directory and build there.")
endif ()

include(GNUInstallDirs)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

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

option(ALUMINUM_ENABLE_DEBUG
  "Enable standard debugging sanity checks."
  OFF)
option(ALUMINUM_ENABLE_DEBUG_HANG_CHECK
  "Enable hang checking."
  OFF)
option(ALUMINUM_ENABLE_SIGNAL_HANDLER
  "Enable Aluminum's bad signal handler (you may suffer)"
  ON)
option(ALUMINUM_ENABLE_NVPROF
  "Enable profiling via nvprof/NVTX."
  OFF)
option(ALUMINUM_ENABLE_ROCTRACER
  "Enable profiling via rocprof/roctx."
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
option(ALUMINUM_ENABLE_HWLOC
  "Enable hwloc"
  ON)

option(ALUMINUM_MPI_SERIALIZE
  "Serialize MPI operations."
  OFF)
option(ALUMINUM_DISABLE_BACKGROUND_STREAMS
  "Never use background streams for non-blocking operations"
  OFF)

option(ALUMINUM_ENABLE_TESTS
  "Build tests."
  OFF)
option(ALUMINUM_ENABLE_BENCHMARKS
  "Build benchmarks."
  OFF)
option(ALUMINUM_LIMIT_TEST_DATATYPES
  "Only support testing/benchmarking with a limited number of datatypes."
  OFF)

# Tuning parameters (in the order they appear in the file). Recall:
# Cache values previously set are not modified. These only take effect
# if the cache values do not already exist.
#
# See extended documentation in cmake/tuning_params.hpp.in.
set(AL_PE_NUM_CONCURRENT_OPS 4
  CACHE STRING
  "Number of concurrent operations the progress engine will perform")

set(AL_PE_NUM_STREAMS 64
  CACHE STRING
  "Max number of streams the progress engine supports")

set(AL_PE_NUM_PIPELINE_STAGES 2
  CACHE STRING
  "Max number of pipeline stages the progress engine supports")

set(AL_PE_INPUT_QUEUE_SIZE 8192
  CACHE STRING
  "Max number of entries in each stream's input queue")

option(AL_PE_ADD_DEFAULT_STREAM
  "Automatically add a default stream entry form the progress engine"
  OFF)

option(AL_PE_STREAM_QUEUE_CACHE
  "Use thread-local cache to map streams to input queues"
  OFF)

option(AL_PE_START_ON_DEMAND
  "Delay starting the progress engine until needed"
  ON)

set(AL_SYNC_MEM_PREALLOC 1024
  CACHE STRING
  "Amount of sync object memory to preallocate in the pool")

set(AL_DEFAULT_CACHE_LINE_SIZE 64) # x86_64
if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^ppc")
  set(AL_DEFAULT_CACHE_LINE_SIZE 128) # power
endif ()
# TODO: I'd like to also detect A64FX but fugaku head nodes are x86
# and cross-compiles are wonky and that's too much work. For now, just
# manually set this on the command line on A64FX.
set(AL_CACHE_LINE_SIZE ${AL_DEFAULT_CACHE_LINE_SIZE}
  CACHE STRING
  "Cache line size in bytes (x86: 64; POWER: 128; A64FX: 256)")

set(AL_DESTRUCTIVE_INTERFERENCE_SIZE 128
  CACHE STRING
  "Minimum size in bytes to avoid destructive interference")

set(AL_CUDA_STREAM_POOL_SIZE 1
  CACHE STRING
  "Number of CUDA streams in the default stream pool")

# END Tuning parameters

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
endif ()
if (ALUMINUM_ENABLE_DEBUG)
  set(AL_DEBUG ON)
endif ()
if (ALUMINUM_ENABLE_DEBUG_HANG_CHECK)
  set(AL_DEBUG_HANG_CHECK ON)
endif ()
if (ALUMINUM_ENABLE_SIGNAL_HANDLER)
  set(AL_SIGNAL_HANDLER ON)
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
if (ALUMINUM_ENABLE_HWLOC)
  set(AL_USE_HWLOC ON)
endif ()
if (ALUMINUM_MPI_SERIALIZE)
  set(AL_MPI_SERIALIZE ON)
endif ()
if (ALUMINUM_DISABLE_BACKGROUND_STREAMS)
  set(AL_DISABLE_BACKGROUND_STREAMS ON)
endif ()
if (ALUMINUM_ENABLE_TESTS)
  set(AL_ENABLE_TESTS ON)
endif ()
if (ALUMINUM_ENABLE_BENCHMARKS)
  set(AL_ENABLE_BENCHMARKS ON)
endif ()
if (ALUMINUM_LIMIT_TEST_DATATYPES)
  set(AL_LIMIT_TEST_DATATYPES ON)
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
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og -gdwarf-4")
endif ()

if (NOT DEFINED BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ON)
endif ()

# Dependencies

find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
if (MPI_CXX_VERSION VERSION_GREATER_EQUAL 4.0)
  message(STATUS "MPI version >= 4.0, enabling large-count MPI support")
  set(AL_HAS_LARGE_COUNT_MPI ON)
else ()
  message(STATUS "MPI version < 4.0, disabling large-count MPI support")
  set(AL_HAS_LARGE_COUNT_MPI OFF)
endif ()
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (ALUMINUM_ENABLE_CALIPER)
  find_package(caliper REQUIRED)
  set(AL_HAS_CALIPER ON)
endif ()

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

if (ALUMINUM_ENABLE_HWLOC)
  if (ALUMINUM_ENABLE_ROCM)
    set(AL_HWLOC_MINIMUM_VERSION "2.3.0")
    find_package(HWLOC "${AL_HWLOC_MINIMUM_VERSION}" REQUIRED)
  else ()
    find_package(HWLOC REQUIRED)
  endif ()
endif ()

if (ALUMINUM_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit 11.0 REQUIRED)

  # If the previous find_package fails, we won't get here.
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

  # From the CUDAToolkit module, we use the following IMPORTED targets:
  #   - CUDA::cudart
  #   - CUDA::cuda_driver
  #   - CUDA::nvToolsExt

  if (ALUMINUM_ENABLE_NCCL)
    find_package(NCCL 2.14.0 REQUIRED)
    set(AL_HAS_NCCL TRUE)
  endif ()

  if (ALUMINUM_ENABLE_NVPROF)
    if (NOT TARGET CUDA::nvToolsExt)
      find_package(NVTX REQUIRED)
      set(AL_HAS_EXTERNAL_NVTX TRUE)
    endif ()
    set(AL_HAS_NVPROF ON)
  endif ()

  # Check that a backend is found
  if (NOT AL_HAS_NCCL AND NOT AL_HAS_MPI_CUDA AND NOT AL_HAS_HOST_TRANSFER)
    message(FATAL_ERROR "CUDA support has been requested and detected. "
      "However, no backend was enabled.")
    set(ALUMINUM_ENABLE_CUDA FALSE)
    set(AL_HAS_CUDA FALSE)
  endif ()

endif (ALUMINUM_ENABLE_CUDA)

if (ALUMINUM_ENABLE_ROCM)
  # NOTE: This MUST come first. Even before the HIP language is
  # enabled. If it is not, then the header guard in
  # AMDDeviceLibsConfig.cmake will block future calls from finding the
  # correct AMD_DEVICE_LIBS_PREFIX, which will cause the command line
  # to have an invalid argument.
  find_package(hip 6.0 CONFIG QUIET)
  if (NOT hip_FOUND)
    find_package(hip 5.0 CONFIG QUIET)
  endif ()
  if (NOT hip_FOUND)
    message(FATAL_ERROR "Failed to find a compatible HIP version (5 or 6).")
  endif ()

  # Now that that's handled, we can enable HIP as a language.
  enable_language(HIP)

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

  # Provides hip_add_executable
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
    find_package(rccl 2.14.0 CONFIG QUIET
      HINTS ${RCCL_DIR} $ENV{RCCL_DIR} ${AL_ROCM_PATH}
      PATH_SUFFIXES lib64/cmake/rccl lib/cmake/rccl
      NO_DEFAULT_PATH)
    find_package(rccl CONFIG REQUIRED)

    find_library(AL_LIB_RT rt)

    message(STATUS "Found RCCL: ${rccl_DIR}")
    set(AL_HAS_NCCL TRUE)
  endif ()

  # Find CUB
  find_package(hipcub CONFIG REQUIRED)

  # (trb 01/03/2022): This is needed for HWLOC stuff to work correctly.
  find_package(rocm_smi CONFIG REQUIRED)

  if (ALUMINUM_ENABLE_ROCTRACER)
    find_package(Roctracer MODULE COMPONENTS roctx)
    set(AL_HAS_ROCTRACER ${Roctracer_FOUND})
  endif ()

endif ()

# Pull the Git submodules.
# Adapted from:
# https://cliutils.gitlab.io/modern-cmake/chapters/projects/submodule.html
find_package(Git QUIET)
if (GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE GIT_SUBMOD_RESULT)
  if (NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR
      "git submodule update --init failed with ${GIT_SUBMOD_RESULT}. "
      "Please checkout submodules")
  endif ()
endif ()

# Build library and executables

# Write the configuration file
# Update to include other profiling interfaces as needed:
if (AL_HAS_NVPROF)
  set(AL_HAS_PROF ON)
endif ()
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/Al_config.hpp.in"
  "${CMAKE_BINARY_DIR}/Al_config.hpp" @ONLY)

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/tuning_params.hpp.in"
  "${CMAKE_BINARY_DIR}/aluminum/tuning_params.hpp" @ONLY)

# Macro for setting full paths to source files.
macro(set_source_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    list(APPEND __tmp_names "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

add_subdirectory(include/aluminum)

# Add in the master header
list(APPEND ALUMINUM_HEADERS "${PROJECT_SOURCE_DIR}/include/Al.hpp")

# Add the library targets
add_subdirectory(src)
add_subdirectory(util)

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
  ${ALUMINUM_VERSION} COMPATIBILITY SameMajorVersion )

set(INCLUDE_INSTALL_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
set(LIB_INSTALL_DIR src)
set(CMAKE_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_MODULE_LOCATION "${PROJECT_SOURCE_DIR}/cmake")
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
install(FILES
  "${CMAKE_BINARY_DIR}/aluminum/tuning_params.hpp"
  DESTINATION ${INCLUDE_INSTALL_DIRS}/aluminum)

# Install the CMake modules we need
install(FILES
  cmake/FindHWLOC.cmake
  cmake/FindNCCL.cmake
  cmake/FindRoctracer.cmake
  DESTINATION ${CMAKE_INSTALL_DIR})
