set_source_path(THIS_DIR_HEADERS
  locked_resource_pool.hpp
  utils.hpp
  )

set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
