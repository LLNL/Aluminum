# This is a simple target. Just needs Al_config.hpp in its include path.
add_executable(al_info al_info.cpp)
target_include_directories(al_info PRIVATE ${CMAKE_BINARY_DIR})
install(
  TARGETS al_info
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
