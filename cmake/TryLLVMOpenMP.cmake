# This file is only going to do OpenMP_CXX. It can be modified later
# if, e.g., OpenMP_C is required.

include(CheckCXXSourceCompiles)

find_library(_OpenMP_LIBRARY
  NAMES omp gomp iomp5md
  HINTS ${OpenMP_DIR} $ENV{OpenMP_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The libomp library.")
find_library(_OpenMP_LIBRARY
  NAMES omp gomp iomp5md)
mark_as_advanced(_OpenMP_LIBRARY)

if (NOT _OpenMP_LIBRARY)
  message(FATAL_ERROR "No OpenMP library found.")
else ()

  get_filename_component(_OpenMP_LIB_DIR "${_OpenMP_LIBRARY}" DIRECTORY)
  
  if (${_OpenMP_LIBRARY} MATCHES "libomp")
    set(OpenMP_libomp_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP omp library.")
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
    set(OpenMP_omp_LIBRARY "${_OpenMP_LIBRARY}")
  elseif (${_OpenMP_LIBRARY} MATCHES "libgomp")
    set(OpenMP_libgomp_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP gomp library.")
    set(OpenMP_CXX_LIB_NAMES gomp)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_gomp_LIBRARY "${_OpenMP_LIBRARY}")
  elseif (${_OpenMP_LIBRARY} MATCHES "libiomp5md")
    set(OpenMP_libiomp5md_LIBRARY ${_OpenMP_LIBRARY}
      CACHE PATH "The OpenMP iomp5md library.")
    set(OpenMP_CXX_LIB_NAMES iomp5md)
    set(OpenMP_CXX_FLAGS "-fopenmp=libiomp5")
    set(OpenMP_iomp5md_LIBRARY "${_OpenMP_LIBRARY}")
  endif ()

  # Let's try this again
  find_package(OpenMP COMPONENTS CXX)
  if (OpenMP_CXX_FOUND)
    if (CMAKE_VERSION VERSION_GREATER 3.13.0)
      target_link_directories(
        OpenMP::OpenMP_CXX INTERFACE "${_OpenMP_LIB_DIR}")
    else ()
      # This isn't great, but it should work. The better solution is
      # to use a version of CMake that is at least 3.13.0.
      set_property(TARGET OpenMP::OpenMP_CXX APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES "-L${_OpenMP_LIB_DIR}")
    endif ()
  endif ()
endif (NOT _OpenMP_LIBRARY)
