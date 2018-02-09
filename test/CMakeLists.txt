# The name changed in CMake 3.10
if (NOT MPIEXEC_EXECUTABLE AND MPIEXEC)
  set(MPIEXEC_EXECUTABLE ${MPIEXEC})
endif ()

add_executable(TestCorrectness.exe
  test_correctness.cpp test_utils.hpp test_utils_cuda.hpp)
target_link_libraries(TestCorrectness.exe PRIVATE aluminum)

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
  test_multi_nballreduces.cpp test_utils.hpp)
target_link_libraries(TestMultiNBAllReduces.exe PRIVATE aluminum)

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
