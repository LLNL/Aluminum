set_full_path(THIS_DIR_HEADERS
  allgather.hpp
  allreduce.hpp
  alltoall.hpp
  bcast.hpp
  communicator.hpp
  gather.hpp
  pt2pt.hpp
  reduce.hpp
  reduce_scatter.hpp
  scatter.hpp
  )

# Custom install
install(FILES ${THIS_DIR_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/ht)

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
