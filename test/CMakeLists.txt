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

set(TEST_HDRS
  ${TEST_BASE_HDRS}
  ${TEST_CUDA_HDRS}
  ${TEST_MPI_CUDA_HDRS}
  ${TEST_HOST_TRANSFER_HDRS}
  ${TEST_NCCL_HDRS})

# These headers are used in the benchmark/ directory
add_library(aluminum_test_headers INTERFACE)

if (AL_HAS_ROCM)
  hipify_header_files(TEST_HEADERS ${TEST_HDRS})
  add_custom_target(gen_test_headers
    DEPENDS ${TEST_HEADERS}
    COMMENT "Generating hipified test headers.")
  add_dependencies(aluminum_test_headers gen_test_headers)
  target_include_directories(
    aluminum_test_headers INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")
else ()
  set(TEST_HEADERS ${TEST_HDRS})
  target_sources(aluminum_test_headers INTERFACE "${TEST_HEADERS}")
  target_include_directories(
    aluminum_test_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
endif ()

if (AL_ENABLE_TESTS)
  set_source_path(TEST_SRCS
    test_ops.cpp
    test_exchange.cpp
    )

  if (AL_HAS_ROCM)
    hipify_source_files(AL_TEST_SOURCES ${TEST_SRCS})
  else ()
    set(AL_TEST_SOURCES ${TEST_SRCS})
  endif ()

  foreach(src ${AL_TEST_SOURCES})
    string(REPLACE ".cpp" ".exe" _tmp_test_exe_name "${src}")
    get_filename_component(_test_exe_name "${_tmp_test_exe_name}" NAME)
    add_executable(${_test_exe_name} ${src})
    target_include_directories(
      ${_test_exe_name} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)
    target_link_libraries(${_test_exe_name} PRIVATE Al aluminum_test_headers)
    if (AL_HAS_CUDA AND NOT AL_HAS_ROCM)
      target_link_libraries(${_test_exe_name} PUBLIC cuda)
    endif()
    target_link_libraries(${_test_exe_name} PRIVATE Threads::Threads)
  endforeach()

  # FIXME: Should any of these tests be enabled with ROCm?
  if (AL_HAS_CUDA AND NOT AL_HAS_ROCM)
    add_executable(test_stream_mem_ops.exe
      test_stream_mem_ops.cpp ${TEST_HEADERS})
    target_link_libraries(test_stream_mem_ops.exe PRIVATE Al)
    target_link_libraries(test_stream_mem_ops.exe PUBLIC cuda)
    target_link_libraries(${_test_exe_name} PRIVATE Threads::Threads)
  endif ()

  if (AL_HAS_MPI_CUDA_RMA AND NOT AL_HAS_ROCM)
    add_executable(test_rma_ring.exe test_rma_ring.cpp ${TEST_HEADERS})
    target_include_directories(
      test_rma_ring.exe SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)
    target_link_libraries(test_rma_ring.exe PRIVATE Al)
    target_link_libraries(test_rma_ring.exe PUBLIC cuda)
    add_executable(test_rma_halo_exchange.exe
      test_rma_halo_exchange.cpp ${TEST_HEADERS})
    target_include_directories(
      test_rma_halo_exchange.exe SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)
    target_link_libraries(test_rma_halo_exchange.exe PRIVATE Al)
    target_link_libraries(test_rma_halo_exchange.exe PUBLIC cuda)
    target_link_libraries(${_test_exe_name} PRIVATE Threads::Threads)
  endif ()
endif ()
