set(BENCHMARK_SRCS
  benchmark_allreduces.cpp
  benchmark_nballreduces.cpp
  benchmark_reductions.cpp)

foreach(src ${BENCHMARK_SRCS})
  string(REPLACE ".cpp" ".exe" _benchmark_exe_name "${src}")
  add_executable(${_benchmark_exe_name} ${src})
  target_link_libraries(${_benchmark_exe_name}
    PRIVATE Al aluminum_test_headers)
  if (AL_HAS_CUDA)
    target_link_libraries(${_benchmark_exe_name} PUBLIC cuda)
  endif ()
  target_include_directories(
    ${_benchmark_exe_name} PRIVATE ${CMAKE_SOURCE_DIR}/test)
endforeach()

add_executable(benchmark_overlap.exe benchmark_overlap.cpp
  $<$<BOOL:${AL_HAS_CUDA}>:wait.cu>)
target_link_libraries(benchmark_overlap.exe PRIVATE Al aluminum_test_headers)
if (AL_HAS_CUDA)
  target_link_libraries(benchmark_overlap.exe PUBLIC cuda)
endif ()
target_include_directories(benchmark_overlap.exe PRIVATE ${CMAKE_SOURCE_DIR}/test)

add_executable(benchmark_pingpong.exe benchmark_pingpong.cpp
  $<$<BOOL:${AL_HAS_CUDA}>:wait.cu>)
target_link_libraries(benchmark_pingpong.exe PRIVATE Al aluminum_test_headers)
if (AL_HAS_CUDA)
  target_link_libraries(benchmark_pingpong.exe PUBLIC cuda)
endif ()
target_include_directories(benchmark_pingpong.exe PRIVATE ${CMAKE_SOURCE_DIR}/test)

if (AL_HAS_CUDA)
  add_executable(benchmark_events.exe benchmark_events.cpp wait.cu)
  target_link_libraries(benchmark_events.exe PRIVATE Al aluminum_test_headers)
  target_link_libraries(benchmark_events.exe PUBLIC cuda)
  target_include_directories(
    benchmark_events.exe PRIVATE ${CMAKE_SOURCE_DIR}/test)

  add_executable(benchmark_waits.exe benchmark_waits.cpp wait.cu)
  target_link_libraries(benchmark_waits.exe PRIVATE Al aluminum_test_headers)
  target_link_libraries(benchmark_waits.exe PUBLIC cuda)
  target_include_directories(
    benchmark_waits.exe PRIVATE ${CMAKE_SOURCE_DIR}/test)
endif ()
