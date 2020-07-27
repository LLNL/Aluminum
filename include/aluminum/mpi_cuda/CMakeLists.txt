set_source_path(THIS_DIR_HEADERS
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
  util.hpp
  )

set_source_path(THIS_DIR_CUDA_RMA_HEADERS
  rma.hpp
  rma_ipc.hpp
  rma_null.hpp
  rma_self.hpp
  )

if (AL_HAS_MPI_CUDA_RMA)
  list(APPEND THIS_DIR_HEADERS "${THIS_DIR_CUDA_RMA_HEADERS}")
endif ()

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
