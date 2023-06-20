set_source_path(AL_BENCHMARK_HEADERS
  benchmark_utils.hpp
  benchmark_utils_mpi.hpp
  benchmark_utils_nccl.hpp
  benchmark_utils_ht.hpp
  wait.hpp)

set_source_path(AL_BENCHMARK_SOURCES
  benchmark_ops.cpp
  bandwidth.cpp)

if (AL_HAS_CUDA OR AL_HAS_ROCM)
  set_source_path(AL_GPU_BENCHMARK_SOURCES
    benchmark_waits.cpp
    benchmark_events.cpp)

  set_source_path(AL_GPU_BENCHMARK_HELPER_SOURCES
    wait.cu
    wait.hpp)
endif ()

foreach(src IN LISTS AL_BENCHMARK_SOURCES AL_GPU_BENCHMARK_SOURCES)
  string(REPLACE ".cpp" "" _tmp_benchmark_exe_name "${src}")
  get_filename_component(_benchmark_exe_name
    "${_tmp_benchmark_exe_name}" NAME)
  add_executable(${_benchmark_exe_name} ${src} ${AL_BENCHMARK_HEADERS})

  # Get the test headers
  target_link_libraries(${_benchmark_exe_name}
    PRIVATE Al aluminum_test_headers)
  target_include_directories(
    ${_benchmark_exe_name} SYSTEM PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)

  # FIXME: Hopefully this can be removed in a future version of ROCm.
  if (AL_HAS_ROCM AND AL_BUILD_TYPE_UPPER MATCHES "DEBUG")
    target_compile_options(${_benchmark_exe_name} PRIVATE "-O0")
  endif ()
endforeach()

# Handle the GPU-specific ones
foreach (src ${AL_GPU_BENCHMARK_SOURCES})
  string(REPLACE ".cpp" "" _tmp_benchmark_exe_name "${src}")
  get_filename_component(_benchmark_exe_name
    "${_tmp_benchmark_exe_name}" NAME)
  target_sources(${_benchmark_exe_name}
    PUBLIC "${AL_GPU_BENCHMARK_HELPER_SOURCES}")
endforeach ()

# Get languages right
if (AL_HAS_ROCM)
  list(FILTER AL_GPU_BENCHMARK_HELPER_SOURCES INCLUDE REGEX "cu$")
  set_source_files_properties(${AL_GPU_BENCHMARK_HELPER_SOURCES}
    PROPERTIES LANGUAGE HIP)
endif ()
