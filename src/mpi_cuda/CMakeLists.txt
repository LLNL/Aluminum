set_full_path(THIS_DIR_HEADERS
  allreduce.hpp
  pt2pt.hpp
  allreduce_ring.hpp
  cuda_kernels.hpp
  cuda_local_transfer.hpp
  util.hpp
  )

set_full_path(THIS_DIR_CUDA_SOURCES
  cuda_kernels.cu
  )

# Custom install
install(FILES ${THIS_DIR_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/mpi_cuda)

# Propagate the files up the tree
set(CUDA_SOURCES ${CUDA_SOURCES} ${THIS_DIR_CUDA_SOURCES} PARENT_SCOPE)
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
