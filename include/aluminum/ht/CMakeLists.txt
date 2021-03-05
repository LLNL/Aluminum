set_source_path(THIS_DIR_HEADERS
  allgather.hpp
  allgatherv.hpp
  allreduce.hpp
  alltoall.hpp
  alltoallv.hpp
  barrier.hpp
  base_state.hpp
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
  )

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
