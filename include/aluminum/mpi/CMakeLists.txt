set_source_path(THIS_DIR_HEADERS
  allgather.hpp
  allgatherv.hpp
  allreduce.hpp
  alltoall.hpp
  alltoallv.hpp
  base_state.hpp
  barrier.hpp
  bcast.hpp
  communicator.hpp
  gather.hpp
  gatherv.hpp
  reduce.hpp
  reduce_scatter.hpp
  reduce_scatterv.hpp
  scatter.hpp
  scatterv.hpp
  pt2pt.hpp
  utils.hpp
  )

set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
