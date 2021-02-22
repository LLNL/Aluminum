# This file provides two functions (hipify_source_files and hipify_header_files)
# for projects that would rather stay primarily in CUDA-land

# hipify_files_internal (OUTPUT_VAR, ARGN)
#
# arguments: output list of processed filenames, files to process...
# outputs: list of build-directory filenames, written to new_filenames
#
# Don't use this one! Use either hipify_source_files or hipify_header_files.
# Their implementation is identical but for an extra extension added to sources.
#
# This function runs hipify-perl on every filename given in ARGN as a build step
# and writes the output to the same place as the initial file, but in the build
# directory.

find_program(HIPIFY_PERL hipify-perl)
if (NOT HIPIFY_PERL)
  message(FATAL_ERROR "hipify-perl was not found. "
    "Please make sure it's in $PATH")
else ()
  message(STATUS "hipify-perl found at ${HIPIFY_PERL}")
endif ()

function (hipify_files_internal OUTPUT_VAR)
  set(_new_filenames "")

  foreach (filename_in ${ARGN})
    string(REPLACE "${PROJECT_SOURCE_DIR}/" "" filename "${filename_in}")
    set(input "${PROJECT_SOURCE_DIR}/${filename}")
    get_filename_component(_tmp_extension "${filename}" EXT)
    if (_tmp_extension STREQUAL ".hpp")
      # Don't add extra ".hpp" for headers.
      set(_tmp_extension)
    endif ()
    set(output "${CMAKE_BINARY_DIR}/${filename}${_tmp_extension}")

    #message(DEBUG "***** Processing ${filename} into ${output}")
    add_custom_command(
      OUTPUT ${output}
      COMMAND ${HIPIFY_PERL} ${input} > ${output}
      DEPENDS ${input}
      VERBATIM)

    list(APPEND _new_filenames ${output})
  endforeach()

  set(${OUTPUT_VAR} ${_new_filenames} PARENT_SCOPE)
endfunction()

function(hipify_source_files OUTPUT_VAR)
  hipify_files_internal(${OUTPUT_VAR} ${ARGN})
  set(${OUTPUT_VAR} ${${OUTPUT_VAR}} PARENT_SCOPE)
endfunction()

function(hipify_header_files OUTPUT_VAR)
  hipify_files_internal(${OUTPUT_VAR} ${ARGN})
  set(${OUTPUT_VAR} ${${OUTPUT_VAR}} PARENT_SCOPE)
endfunction()
