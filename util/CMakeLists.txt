set_source_path(UTIL_SRCS
  al_info.cpp)

set_source_path(UTIL_HDRS
)

if (AL_HAS_ROCM)
  hipify_header_files(AL_UTIL_HEADERS ${UTIL_HDRS})
  hipify_source_files(AL_UTIL_SOURCES ${UTIL_SRCS})
else ()
  set(AL_UTIL_HEADERS ${UTIL_HDRS})
  set(AL_UTIL_SOURCES ${UTIL_SRCS})
endif ()

foreach (src ${AL_UTIL_SOURCES})
  string(REPLACE ".cpp" "" _tmp_exe_name "${src}")
  get_filename_component(_exe_name "${_tmp_exe_name}" NAME)
  if (AL_HAS_ROCM)
    add_executable(${_exe_name} ${src} ${AL_UTIL_HEADERS})
    target_include_directories(${_exe_name} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  else ()
    add_executable(${_exe_name} ${src})
  endif ()

  target_link_libraries(${_exe_name} PRIVATE Al)

  # Use more caution now that "AL_HAS_CUDA" is overloaded.
  if (AL_HAS_CUDA AND NOT AL_HAS_ROCM)
    target_link_libraries(${_exe_name} PUBLIC cuda)
  endif ()

  # FIXME: Hopefully this can be removed in a future version of ROCm.
  if (AL_HAS_ROCM AND AL_BUILD_TYPE_UPPER MATCHES "DEBUG")
    target_compile_options(${_exe_name} PRIVATE "-O0")
  endif ()

  install(
    TARGETS ${_exe_name}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endforeach ()
