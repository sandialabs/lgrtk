cmake_minimum_required(VERSION 3.0.0)

project(LGR VERSION 1.0.0 LANGUAGES CXX)

include(bob.cmake)

bob_begin_package()

set(LGR_USE_Omega_h_DEFAULT ON)
bob_add_dependency(PUBLIC NAME Omega_h TARGETS Omega_h::omega_h)
set(LGR_USE_GTest_DEFAULT ${BUILD_TESTING})
bob_add_dependency(PUBLIC NAME GTest TARGETS GTest::gtest_main)
set(LGR_USE_SEACASAprepro_DEFAULT ON)
bob_add_dependency(PUBLIC NAME SEACASAprepro TARGETS aprepro_lib)

set(CMAKE_CXX_STANDARD "11")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

bob_input(LGR_ELEMENTS "Bar2;Tri3;Tet4" STRING "Element types to instantiate")

foreach(ELEMENT IN LISTS LGR_ELEMENTS)
  if(${ELEMENT} STREQUAL "Bar2")
    set(LGR_BAR2 ON)
  endif()
  if(${ELEMENT} STREQUAL "Tri3")
    set(LGR_TRI3 ON)
  endif()
  if(${ELEMENT} STREQUAL "Tri6")
    set(LGR_TRI6 ON)
  endif()
  if(${ELEMENT} STREQUAL "Quad4")
    set(LGR_QUAD4 ON)
  endif()
  if(${ELEMENT} STREQUAL "Tet4")
    set(LGR_TET4 ON)
  endif()
  if(${ELEMENT} STREQUAL "CompTet")
    set(LGR_COMPTET ON)
  endif()
endforeach()

set(CMAKE_CXX_FLAGS "${Omega_h_CXX_FLAGS}")

find_program(LGR_CUBIT NAMES cubit cubit.command PATHS "${CUBIT_PREFIX}")
if (LGR_CUBIT)
  message(STATUS "Found LGR_CUBIT=\"${LGR_CUBIT}\"")
  set(LGR_USE_CUBIT ON)
else()
  set(LGR_USE_CUBIT OFF)
endif()

set(LGR_KEY_BOOLS
  LGR_USE_CUBIT
  LGR_USE_GTest
  LGR_BAR2
  LGR_TRI3
  LGR_TRI6
  LGR_QUAD4
  LGR_TET4
  LGR_COMPTET
  )

set(LGR_KEY_STRINGS
  LGR_CUBIT
  )

add_subdirectory(src)
if (BUILD_TESTING)
  if (LGR_USE_GTest)
    add_subdirectory(unit_tests)
  endif()
  add_subdirectory(inputs)
endif()

bob_end_package()
