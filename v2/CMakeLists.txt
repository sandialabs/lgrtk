cmake_minimum_required(VERSION 3.0.0)

project(LGR VERSION 1.0.0 LANGUAGES CXX)

include(bob.cmake)

bob_begin_package()

set(LGR_USE_Omega_h_DEFAULT ON)
bob_add_dependency(PUBLIC NAME Omega_h TARGETS Omega_h::omega_h)
set(LGR_USE_GTest_DEFAULT ${BUILD_TESTING})
bob_add_dependency(PUBLIC NAME GTest TARGETS GTest::gtest_main)

set(CMAKE_CXX_FLAGS "${Omega_h_CXX_FLAGS}")

add_subdirectory(src)
if (BUILD_TESTING)
  add_subdirectory(unit_tests)
  add_subdirectory(inputs)
endif()

bob_end_package()
