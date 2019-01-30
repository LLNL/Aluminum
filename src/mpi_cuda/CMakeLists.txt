set_full_path(THIS_DIR_HEADERS
  allreduce.hpp
  allreduce_ring.hpp
  communicator.hpp
  cuda_kernels.hpp
  cuda_local_transfer.hpp
  util.hpp
  )

set_full_path(THIS_DIR_CXX_SOURCES
  communicator.cpp
  )

if (AL_HAS_MPI_CUDA_RMA)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/rma.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rma_ipc.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rma_null.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rma_self.hpp")
  list(APPEND THIS_DIR_CXX_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/rma.cpp")
endif ()

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
