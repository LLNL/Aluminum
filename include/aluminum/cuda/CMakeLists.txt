set_source_path(THIS_DIR_HEADERS
  cuda.hpp
  cuda_mempool.hpp
  events.hpp
  gpu_status_flag.hpp
  gpu_wait.hpp
  helper_kernels.hpp
  streams.hpp
  sync_memory.hpp
  )

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
