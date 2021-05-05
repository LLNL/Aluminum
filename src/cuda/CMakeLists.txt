set_source_path(THIS_DIR_CUDA_CXX_SOURCES
  cuda.cpp
  gpu_status_flag.cpp
  gpu_wait.cpp
  streams.cpp
  )
set_source_path(THIS_DIR_CUDA_SOURCES
  helper_kernels.cu
  )

if (AL_HAS_CUDA)
  list(APPEND THIS_DIR_CXX_SOURCES "${THIS_DIR_CUDA_CXX_SOURCES}")
endif ()

set(CXX_SOURCES "${CXX_SOURCES}" "${THIS_DIR_CXX_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CUDA_SOURCES}" PARENT_SCOPE)
