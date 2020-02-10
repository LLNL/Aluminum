# This file provides two functions (hipify_source_files and hipify_header_files)
# for projects that would rather stay primarily in CUDA-land

# hipify_files_internal (extension, new_filenames, ARGN)
#
# arguments: extension to give the target file, output list of processed
#            filenames, files to process...
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
    message(FATAL_ERROR "hipify-perl was not found. Please make sure it's in $PATH")
else ()
    message(STATUS "hipify-perl found at ${HIPIFY_PERL}")
endif ()

function (hipify_files_internal extension new_filenames)
  set(${new_filenames} "")

  foreach (filename ${ARGN})
    set(input "${PROJECT_SOURCE_DIR}/${filename}")
    set(output "${CMAKE_BINARY_DIR}/${filename}${extension}")

    message(DEBUG "Processing ${filename} into ${CMAKE_BINARY_DIR}/${filename}${extension}")

    add_custom_command(
      OUTPUT ${output}
      COMMAND ${HIPIFY_PERL} ${input} > ${output}
      DEPENDS ${input}
      VERBATIM)

    list(APPEND ${new_filenames} ${output})
  endforeach()

  set(${new_filenames} ${${new_filenames}} PARENT_SCOPE)
endfunction()

function(hipify_source_files new_filenames)
  hipify_files_internal(".cpp" ${new_filenames} ${ARGN})
  set(${new_filenames} ${${new_filenames}} PARENT_SCOPE)
endfunction()

function(hipify_header_files new_filenames)
  hipify_files_internal("" ${new_filenames} ${ARGN})
  set(${new_filenames} ${${new_filenames}} PARENT_SCOPE)
endfunction()
