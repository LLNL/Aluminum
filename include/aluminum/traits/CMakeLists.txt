set_source_path(THIS_DIR_HEADERS
  ht_traits.hpp
  mpi_traits.hpp
  nccl_traits.hpp
  traits_base.hpp
  traits.hpp
)

set(THIS_DIR_HEADERS_TO_INSTALL ${THIS_DIR_HEADERS})

set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
