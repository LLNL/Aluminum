set(BENCHMARK_SRCS
  benchmark_allreduces.cpp
  benchmark_nballreduces.cpp
  benchmark_overlap.cpp
  benchmark_reductions.cpp)

foreach(src ${BENCHMARK_SRCS})
  string(REPLACE ".cpp" ".exe" _benchmark_exe_name "${src}")
  add_executable(${_benchmark_exe_name} ${src})
  target_link_libraries(${_benchmark_exe_name}
    PRIVATE aluminum aluminum_test_headers)
  target_include_directories(
    ${_benchmark_exe_name} PRIVATE ${CMAKE_SOURCE_DIR}/test)
endforeach()
