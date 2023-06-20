set_source_path(TEST_BASE_HDRS
  test_utils.hpp
  test_utils_mpi.hpp
  op_dispatcher.hpp
  op_runner_impl.hpp
  hang_watchdog.hpp)
if (AL_HAS_CUDA)
  set_source_path(TEST_CUDA_HDRS cuda_vector.hpp)
endif ()
if (AL_HAS_MPI_CUDA)
  set_source_path(TEST_MPI_CUDA_HDRS test_utils_mpi_cuda.hpp)
endif ()
if (AL_HAS_HOST_TRANSFER)
  set_source_path(TEST_HOST_TRANSFER_HDRS test_utils_ht.hpp)
endif ()
if (AL_HAS_NCCL)
  set_source_path(TEST_NCCL_HDRS test_utils_nccl.hpp)
endif ()

set(TEST_HEADERS
  ${TEST_BASE_HDRS}
  ${TEST_CUDA_HDRS}
  ${TEST_MPI_CUDA_HDRS}
  ${TEST_HOST_TRANSFER_HDRS}
  ${TEST_NCCL_HDRS})

# These headers are used in the benchmark/ directory
add_library(aluminum_test_headers INTERFACE "${TEST_HEADERS}")
target_include_directories(
  aluminum_test_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# Make a quick exit before we add any tests
if (NOT AL_ENABLE_TESTS)
  return()
endif ()

set_source_path(AL_TEST_SOURCES
  test_ops.cpp
  test_exchange.cpp
)

set_source_path(AL_GPU_ONLY_TEST_SOURCES
  test_stream_mem_ops.cpp
)

if (AL_HAS_CUDA OR AL_HAS_ROCM)
  list(APPEND AL_TEST_SOURCES ${AL_GPU_ONLY_TEST_SOURCES})
endif ()

foreach(src ${AL_TEST_SOURCES})
  string(REPLACE ".cpp" "" _tmp_test_exe_name "${src}")
  get_filename_component(_test_exe_name "${_tmp_test_exe_name}" NAME)
  add_executable(${_test_exe_name} ${src})
  target_include_directories(${_test_exe_name}
    SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)
  target_link_libraries(${_test_exe_name} PRIVATE Al aluminum_test_headers)
  # FIXME: Hopefully this can be removed in a future version of ROCm.
  if (AL_HAS_ROCM AND AL_BUILD_TYPE_UPPER MATCHES "DEBUG")
    target_compile_options(${_test_exe_name} PRIVATE "-O0")
  endif ()
endforeach()

if (AL_HAS_MPI_CUDA_RMA AND NOT AL_HAS_ROCM)
  add_executable(test_rma_ring.exe test_rma_ring.cpp ${TEST_HEADERS})
  target_include_directories(
    test_rma_ring.exe SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)
  target_link_libraries(test_rma_ring.exe PRIVATE Al)
  add_executable(test_rma_halo_exchange.exe
    test_rma_halo_exchange.cpp ${TEST_HEADERS})
  target_include_directories(
    test_rma_halo_exchange.exe SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)
  target_link_libraries(test_rma_halo_exchange.exe PRIVATE Al)
endif ()
