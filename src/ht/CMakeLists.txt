set_full_path(THIS_DIR_HEADERS
  allgather.hpp
  allreduce.hpp
  alltoall.hpp
  bcast.hpp
  communicator.hpp
  gather.hpp
  reduce.hpp
  reduce_scatter.hpp
  scatter.hpp
  pt2pt.hpp
  )

set_full_path(THIS_DIR_CXX_SOURCES
  )

# Custom install
install(FILES ${THIS_DIR_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/ht)

# Propagate the files up the tree
set(CXX_SOURCES ${CXX_SOURCES} ${THIS_DIR_CXX_SOURCES} PARENT_SCOPE)
set(CUDA_SOURCES ${CUDA_SOURCES} ${THIS_DIR_CUDA_SOURCES} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
