#[=============[.rst
FindCUB
==========

Finds the CUB library.

The following variables will be defined::

  CUB_FOUND          - True if the system has the CUB library.
  CUB_INCLUDE_DIRS   - The include directory needed for CUB.

The following cache variable will be set and marked as "advanced"::

  CUB_INCLUDE_DIR - The include directory needed for CUB.

In addition, the :prop_tgt:`IMPORTED` target ``cuda::CUB`` will
be created.

#]=============]


find_path(CUB_INCLUDE_PATH cub/cub.cuh
  HINTS ${CUB_DIR} $ENV{CUB_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The CUB header directory."
  )
find_path(CUB_INCLUDE_PATH cub/cub.cuh)

set(CUB_INCLUDE_DIRS "${CUB_INCLUDE_PATH}")

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUB
  DEFAULT_MSG CUB_INCLUDE_PATH)

# Setup the imported target
if (NOT TARGET cuda::CUB)
  add_library(cuda::CUB INTERFACE IMPORTED)
endif (NOT TARGET cuda::CUB)

# Set the include directories for the target
if (CUB_INCLUDE_PATH)
  set_property(TARGET cuda::CUB
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CUB_INCLUDE_PATH})
endif ()

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE CUB_INCLUDE_PATH)

# Set the libraries
set(CUB_LIBRARIES cuda::CUB)
