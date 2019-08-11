set_full_path(THIS_DIR_HEADERS
  communicator.hpp
  base_state.hpp
  allgather.hpp
  allreduce.hpp
  alltoall.hpp
  bcast.hpp
  gather.hpp
  reduce.hpp
  reduce_scatter.hpp
  scatter.hpp
  pt2pt.hpp
  )

install(FILES ${THIS_DIR_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/cuda_aware_mpi)

set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
