set_full_path(TEST_HEADERS
  test_utils.hpp)
if (AL_HAS_CUDA)
  list(APPEND TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_cuda.hpp)
endif ()

if (AL_HAS_MPI_CUDA)
  list(APPEND TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_mpi_cuda.hpp)
endif ()

if (AL_HAS_NCCL)
  list(APPEND TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_nccl_cuda.hpp)
endif ()

if (AL_HAS_CUDA_AWARE_MPI)
  list(APPEND TEST_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/test_utils_cuda_aware_mpi.hpp)
endif ()

# These headers are used in the benchmark/ directory
add_library(aluminum_test_headers INTERFACE)
target_sources(aluminum_test_headers INTERFACE "${TEST_HEADERS}")
target_include_directories(
  aluminum_test_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

set(TEST_SRCS
  test_allreduce.cpp
  test_reduce.cpp
  test_reduce_scatter.cpp
  test_allgather.cpp
  test_alltoall.cpp
  test_bcast.cpp
  test_gather.cpp
  test_scatter.cpp
  test_pt2pt.cpp
  test_multi_nballreduces.cpp)

foreach(src ${TEST_SRCS})
  string(REPLACE ".cpp" ".exe" _test_exe_name "${src}")
  add_executable(${_test_exe_name} ${src})
  target_link_libraries(${_test_exe_name} PRIVATE Al aluminum_test_headers)
  if (AL_HAS_CUDA)
    target_link_libraries(${_test_exe_name} PUBLIC cuda)
  endif()
endforeach()

if (AL_HAS_CUDA)
  add_executable(test_stream_mem_ops.exe
  test_stream_mem_ops.cpp ${TEST_HEADERS})
  target_link_libraries(test_stream_mem_ops.exe PRIVATE Al)
  target_link_libraries(test_stream_mem_ops.exe PUBLIC cuda)
endif ()

if (AL_HAS_MPI_CUDA_RMA)
  add_executable(test_rma_ring.exe test_rma_ring.cpp ${TEST_HEADERS})
  target_link_libraries(test_rma_ring.exe PRIVATE Al)
  target_link_libraries(test_rma_ring.exe PUBLIC cuda)
  add_executable(test_rma_halo_exchange.exe
    test_rma_halo_exchange.cpp ${TEST_HEADERS})
  target_link_libraries(test_rma_halo_exchange.exe PRIVATE Al)
  target_link_libraries(test_rma_halo_exchange.exe PUBLIC cuda)
endif ()
