set_source_path(BENCHMARK_HDRS
  benchmark_utils.hpp
  benchmark_utils_mpi.hpp
  benchmark_utils_nccl.hpp
  benchmark_utils_ht.hpp
  wait.hpp)

set_source_path(BENCHMARK_SRCS
  benchmark_ops.cpp)

if (AL_HAS_ROCM)
  hipify_header_files(AL_BENCHMARK_HEADERS ${BENCHMARK_HDRS})
  hipify_source_files(AL_BENCHMARK_SOURCES ${BENCHMARK_SRCS})
else ()
  set(AL_BENCHMARK_SOURCES ${BENCHMARK_SRCS})
endif ()

foreach(src ${AL_BENCHMARK_SOURCES})
  string(REPLACE ".cpp" ".exe" _tmp_benchmark_exe_name "${src}")
  get_filename_component(_benchmark_exe_name
    "${_tmp_benchmark_exe_name}" NAME)
  if (AL_HAS_ROCM)
    hip_add_executable(${_benchmark_exe_name} ${src} ${AL_BENCHMARK_HEADERS})
    target_include_directories(${_benchmark_exe_name} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  else ()
    add_executable(${_benchmark_exe_name} ${src})
  endif ()

  # Get the test headers
  target_link_libraries(${_benchmark_exe_name}
    PRIVATE Al aluminum_test_headers)
  target_include_directories(
    ${_benchmark_exe_name} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)

  # Use more caution now that "AL_HAS_CUDA" is overloaded.
  if (AL_HAS_CUDA AND NOT AL_HAS_ROCM)
    target_link_libraries(${_benchmark_exe_name} PUBLIC cuda)
  endif ()
endforeach()

# Do it all again for "special" benchmarks...
set_source_path(SPECIAL_BENCHMARK_BASE_SRCS)
if (AL_HAS_CUDA OR AL_HAS_ROCM)
  set_source_path(SPECIAL_BENCHMARK_WAIT_SRCS wait.cu)
  set_source_path(SPECIAL_BENCHMARK_WAIT_HDRS wait.hpp)

  if (NOT AL_HAS_ROCM)
    # These have some CUDA-specific idiosyncracies that I haven't
    # sorted out yet. So we leave them out for now.
    set_source_path(SPECIAL_BENCHMARK_GPU_SRCS
      benchmark_events.cpp
      benchmark_waits.cpp)
  endif ()
endif ()

if (AL_HAS_ROCM)
  hipify_source_files(AL_SPECIAL_BENCHMARK_SOURCES
    ${SPECIAL_BENCHMARK_BASE_SRCS}
    ${SPECIAL_BENCHMARK_GPU_SRCS})
  hipify_source_files(AL_WAIT_SRCS ${SPECIAL_BENCHMARK_WAIT_SRCS})
  hipify_header_files(AL_WAIT_HDRS ${SPECIAL_BENCHMARK_WAIT_HDRS})
  set(AL_WAIT_SOURCES ${AL_WAIT_SRCS} ${AL_WAIT_HDRS})
  add_custom_target(gen_wait_source_files
    DEPENDS ${AL_WAIT_SOURCES}
    COMMENT "Generating hipified wait.{hpp,cu} files.")
else ()
  set(AL_SPECIAL_BENCHMARK_SOURCES
    ${SPECIAL_BENCHMARK_BASE_SRCS}
    ${SPECIAL_BENCHMARK_GPU_SRCS})
  set(AL_WAIT_SOURCES ${SPECIAL_BENCHMARK_WAIT_SRCS})
endif ()

foreach(src ${AL_SPECIAL_BENCHMARK_SOURCES})
  string(REPLACE ".cpp" ".exe" _tmp_benchmark_exe_name "${src}")
  get_filename_component(_benchmark_exe_name
    "${_tmp_benchmark_exe_name}" NAME)
  if (AL_HAS_ROCM)
    hip_add_executable(${_benchmark_exe_name} ${src} ${AL_WAIT_SOURCES})
    target_include_directories(${_benchmark_exe_name} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
    add_dependencies(${_benchmark_exe_name} gen_wait_source_files)
  else ()
    add_executable(${_benchmark_exe_name} ${src} ${AL_WAIT_SOURCES})
  endif ()

  target_link_libraries(${_benchmark_exe_name}
    PRIVATE Al aluminum_test_headers)
  target_include_directories(
    ${_benchmark_exe_name} SYSTEM PRIVATE ${CMAKE_SOURCE_DIR}/third_party/cxxopts/include)

  if (AL_HAS_CUDA AND NOT AL_HAS_ROCM)
    target_link_libraries(${_benchmark_exe_name} PRIVATE cuda)
  endif ()
endforeach ()
