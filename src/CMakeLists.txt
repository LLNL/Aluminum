set(ALUMINUM_HEADERS
  allreduce.hpp
  allreduce_impl.hpp
  allreduce_mempool.hpp
  allreduce_mpi_impl.hpp
  allreduce_nccl_impl.hpp
  common.h
  tuning_params.hpp
  )
set(ALUMINUM_SOURCES
  allreduce.cpp
  allreduce_mpi_impl.cpp
  )

add_library(aluminum ${ALUMINUM_SOURCES} ${ALUMINUM_HEADERS})
target_include_directories(aluminum PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/aluminum>)
target_link_libraries(aluminum PUBLIC
  MPI::MPI_CXX HWLOC::hwloc OpenMP::OpenMP_CXX
  $<$<BOOL:${ALUMINUM_HAS_CUDA}>:cuda::cuda>)
target_compile_definitions(aluminum PUBLIC
  $<$<BOOL:${ALUMINUM_HAS_CUDA}>:ALUMINUM_HAS_CUDA>
  $<$<BOOL:${ALUMINUM_HAS_NCCL}>:ALUMINUM_HAS_NCCL>)
target_compile_features(aluminum PUBLIC cxx_std_11)

install(TARGETS aluminum
  EXPORT AluminumTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum
  )

install(EXPORT AluminumTargets
  NAMESPACE AL::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/aluminum
  )

install(FILES ${ALUMINUM_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/aluminum)
