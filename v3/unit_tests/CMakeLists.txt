
option(LGR_ENABLE_UNIT_TESTS "Build support for unit tests" ON)

if (LGR_ENABLE_UNIT_TESTS AND NOT LGR_ENABLE_CUDA)

  set(LGR_UNIT_SOURCES
    tensor.cpp
    maxent.cpp
    exodus.cpp
    ut_main.cpp)

  include(CTest)

  find_package(GTest REQUIRED)
  include(GoogleTest)
  add_executable(ut_lgr ${LGR_UNIT_SOURCES})
  set_property(TARGET ut_lgr PROPERTY CXX_STANDARD "14")
  set_property(TARGET ut_lgr PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET ut_lgr PROPERTY CXX_EXTENSIONS OFF)
  gtest_discover_tests(ut_lgr)
  target_include_directories(ut_lgr PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> ${GTEST_INCLUDE_DIR})
  target_link_libraries(ut_lgr PRIVATE "lgrlib;${GTEST_LIBRARY};pthread")
endif()
