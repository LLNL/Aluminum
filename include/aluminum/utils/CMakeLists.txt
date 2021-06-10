set_source_path(THIS_DIR_HEADERS
  caching_allocator.hpp
  locked_resource_pool.hpp
  meta.hpp
  mpsc_queue.hpp
  spsc_queue.hpp
  utils.hpp
  )

set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
