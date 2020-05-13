
option(LGR_ENABLE_UNIT_TESTS "Build support for unit tests" ON)

if (LGR_ENABLE_UNIT_TESTS)
  set(LGR_UNIT_SOURCES
    adapt.cpp
    distances.cpp
    map.cpp
    materials.cpp
    maxent.cpp
    mechanics.cpp
    quaternion.cpp
    tensor.cpp
    vtk.cpp
    ut_main.cpp)

  if (LGR_ENABLE_EXODUS)
    set(LGR_UNIT_SOURCES ${LGR_UNIT_SOURCES} exodus.cpp)
  endif()

  if (LGR_ENABLE_SEARCH)
    set(LGR_UNIT_SOURCES ${LGR_UNIT_SOURCES} search.cpp)
  endif()

  if (LGR_ENABLE_CUDA AND NOT LGR_USE_NVCC_WRAPPER)
    set_source_files_properties(${LGR_UNIT_SOURCES} PROPERTIES LANGUAGE CUDA)
  endif()

  include(CTest)

  find_package(GTest REQUIRED)
  include(GoogleTest)

  add_executable(ut_lgr ${LGR_UNIT_SOURCES})
  if (LGR_ENABLE_CUDA)
    target_compile_definitions(ut_lgr PUBLIC -DLGR_ENABLE_CUDA)
  endif()
  set_property(TARGET ut_lgr PROPERTY CXX_STANDARD "14")
  set_property(TARGET ut_lgr PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET ut_lgr PROPERTY CXX_EXTENSIONS OFF)

  gtest_discover_tests(ut_lgr)
  target_include_directories(ut_lgr PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> ${GTEST_INCLUDE_DIR})
  target_link_libraries(ut_lgr PRIVATE "lgrlib;${GTEST_LIBRARY};pthread")

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/tets.g ${CMAKE_CURRENT_BINARY_DIR}/tets.g COPYONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.g ${CMAKE_CURRENT_BINARY_DIR}/cube.g COPYONLY)
endif()
