set(unit_test_exes
  hyper_ep_unit_tests.cpp
  ideal_gas_unit_tests.cpp
  mie_gruneisen_unit_tests.cpp
  linear_algebra_unit_tests.cpp
  circuit_unit_tests.cpp
  )

if(LGR_COMPTET)
  list(APPEND unit_test_exes comptet_unit_tests.cpp)
endif()

if(LGR_TRI6)
  list(APPEND unit_test_exes tri6_unit_tests.cpp)
endif()

add_executable(unit_tests ${unit_test_exes})
target_link_libraries(unit_tests
    PUBLIC
    lgr_library)
bob_link_dependency(unit_tests PUBLIC GTest)
add_test(NAME run_unit_tests COMMAND
  "${CMAKE_CURRENT_BINARY_DIR}/unit_tests")

bob_end_subdir()
