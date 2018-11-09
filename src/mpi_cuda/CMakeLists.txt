set_full_path(THIS_DIR_HEADERS
  allgather.hpp
  allreduce.hpp
  allreduce_ring.hpp
  alltoall.hpp
  bcast.hpp
  communicator.hpp
  cuda_kernels.hpp
  cuda_local_transfer.hpp
  gather.hpp
  reduce.hpp
  reduce_scatter.hpp
  scatter.hpp
  pt2pt.hpp
  util.hpp
  rma.hpp
  rma_ipc.hpp
  )

set_full_path(THIS_DIR_CXX_SOURCES
  communicator.cpp
  rma.cpp
  )

set_full_path(THIS_DIR_CUDA_SOURCES
  cuda_kernels.cu
  )

# Custom install
install(FILES ${THIS_DIR_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/mpi_cuda)

# Propagate the files up the tree
set(CXX_SOURCES ${CXX_SOURCES} ${THIS_DIR_CXX_SOURCES} PARENT_SCOPE)
set(CUDA_SOURCES ${CUDA_SOURCES} ${THIS_DIR_CUDA_SOURCES} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
