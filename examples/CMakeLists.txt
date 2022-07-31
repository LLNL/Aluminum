cmake_minimum_required(VERSION 3.21)

project(AlExamples
  VERSION 0.0.1
  DESCRIPTION "Examples of how to use Aluminum"
  LANGUAGES CXX
)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Configuration type" FORCE)
endif ()

find_package(Aluminum 1.0.0 REQUIRED)

set(SOURCES
  hello_world.cpp
)

foreach (src ${SOURCES})
  string(REPLACE ".cpp" "" _tmp_exe_name "${src}")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  add_executable(${_exe_name} ${src})
  target_link_libraries(${_exe_name} PUBLIC ${Aluminum_LIBRARIES})
endforeach ()
