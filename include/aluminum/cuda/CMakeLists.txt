set_source_path(THIS_DIR_CUDA_HEADERS
  cuda.hpp
  gpu_status_flag.hpp
  gpu_wait.hpp
  helper_kernels.hpp
  sync_memory.hpp
  )

if (AL_HAS_CUDA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_CUDA_HEADERS}")
endif ()

set(ALUMINUM_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
