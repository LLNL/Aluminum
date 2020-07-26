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
  )

if (AL_HAS_ROCM)
  hipify_header_files(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
else ()
  set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})
endif ()

# Custom install
install(FILES ${THIS_DIR_HEADERS_TO_INSTALL}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum/ht)

# Propagate the files up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
