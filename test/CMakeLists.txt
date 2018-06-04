# The name changed in CMake 3.10
if (NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
  set(MPIEXEC_EXECUTABLE ${MPIEXEC})
endif ()

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

# These headers are used in the benchmark/ directory
add_library(aluminum_test_headers INTERFACE)
target_sources(aluminum_test_headers INTERFACE "${TEST_HEADERS}")
target_include_directories(
  aluminum_test_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(TestCorrectness.exe test_correctness.cpp ${TEST_HEADERS})
target_link_libraries(TestCorrectness.exe PRIVATE Al)

# This is mostly a sanity check
set(TEST_ARGS MPI 8)
add_test(NAME TestCorrectness
  COMMAND $<TARGET_FILE:TestCorrectness.exe> ${TEST_ARGS})

if (MPIEXEC_EXECUTABLE)
  add_test(NAME TestCorrectness_np4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:TestCorrectness.exe>
    ${MPIEXEC_POSTFLAGS} ${TEST_ARGS})
endif ()

add_executable(TestMultiNBAllReduces.exe
  test_multi_nballreduces.cpp ${TEST_HEADERS})
target_link_libraries(TestMultiNBAllReduces.exe PRIVATE Al)

set(TEST_ARGS "8")
add_test(NAME TestMultiNBAllReduces
  COMMAND $<TARGET_FILE:TestMultiNBAllReduces.exe> ${TEST_ARGS})

if (MPIEXEC_EXECUTABLE)
  add_test(NAME TestMultiNBAllReduces_np4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:TestMultiNBAllReduces.exe>
    ${MPIEXEC_POSTFLAGS} ${TEST_ARGS})
endif ()

add_executable(TestNCCLCollectives.exe
  test_nccl_collectives.cpp ${TEST_HEADERS})
target_link_libraries(TestNCCLCollectives.exe PRIVATE Al)

set(TEST_ARGS "8")
add_test(NAME TestNCCLCollectives
  COMMAND $<TARGET_FILE:TestNCCLCollectives.exe> ${TEST_ARGS})

if (MPIEXEC_EXECUTABLE)
  add_test(NAME TestNCCLCollectives_np4
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
    ${MPIEXEC_PREFLAGS}
    $<TARGET_FILE:TestNCCLCollectives.exe>
    ${MPIEXEC_POSTFLAGS} ${TEST_ARGS})
endif ()
