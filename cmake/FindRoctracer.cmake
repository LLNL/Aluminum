# Sets the following variables
#
#   Roctracer_FOUND
#   Roctracer_LIBRARIES
#
# Defines the following imported target:
#
#   roctracer::roctracer
#   roctracer::roctracer_api
#   roctracer::roctx_api
#

set(_supported_components roctracer roctx)
if (NOT Roctracer_FIND_COMPONENTS)
  set(Roctracer_FIND_COMPONENTS ${_supported_components})
endif ()

foreach (comp IN LISTS Roctracer_FIND_COMPONENTS)
  if (NOT ${comp} IN_LIST _supported_components)
    message(FATAL_ERROR
      "Cannot specify component \"${comp}\" for package Roctracer. "
      "Supported components are: ${_supported_components}.")
  endif ()

  set(_header_name "${comp}.h")
  set(_lib_name "${comp}64")

  find_path(${comp}_INCLUDE_PATH ${_header_name}
    HINTS ${ROCM_PATH}/roctracer $ENV{ROCM_PATH}/roctracer
    PATH_SUFFIXES include
    DOC "The ${comp} include directory for roctracer."
    NO_DEFAULT_PATH)
  find_path(${comp}_INCLUDE_PATH ${_header_name}
    HINTS ${ROCM_PATH}/include/roctracer $ENV{ROCM_PATH}/include/roctracer
    DOC "The ${comp} include directory for roctracer."
    NO_DEFAULT_PATH)
  find_path(${comp}_INCLUDE_PATH ${_header_name})

  find_library(${comp}_LIBRARY ${_lib_name}
    HINTS ${ROCM_PATH}/roctracer $ENV{ROCM_PATH}/roctracer
    HINTS ${ROCM_PATH} $ENV{ROCM_PATH}
    PATH_SUFFIXES lib64 lib
    DOC "The ${comp} library for roctracer."
    NO_DEFAULT_PATH)
  find_library(${comp}_LIBRARY ${_lib_name})

  if (${comp}_LIBRARY AND ${comp}_INCLUDE_PATH)
    set(Roctracer_${comp}_FOUND TRUE)

    if (NOT TARGET roctracer::${comp}_api)
      add_library(roctracer::${comp}_api INTERFACE IMPORTED)
    endif ()
    target_link_libraries(roctracer::${comp}_api INTERFACE
      "${${comp}_LIBRARY}")
    target_include_directories(roctracer::${comp}_api INTERFACE
      "${${comp}_INCLUDE_PATH}")

    mark_as_advanced(${comp}_LIBRARY)
    mark_as_advanced(${comp}_INCLUDE_PATH)

    list(APPEND _imported_libraries roctracer::${comp}_api)
  else ()
    set(Roctracer_${comp}_FOUND FALSE)
  endif ()
endforeach ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Roctracer HANDLE_COMPONENTS)

if (Roctracer_FOUND)
  if (NOT TARGET roctracer::roctracer)
    add_library(roctracer::roctracer INTERFACE IMPORTED)
  endif ()
  foreach (lib IN LISTS _imported_libraries)
    target_link_libraries(roctracer::roctracer INTERFACE ${lib})
  endforeach ()
  set(Roctracer_LIBRARIES roctracer::roctracer)
endif (Roctracer_FOUND)
