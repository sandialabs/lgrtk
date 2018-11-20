add_executable(unit_tests
  hyper_ep_unit_tests.cpp
  ideal_gas_unit_tests.cpp
  mie_gruneisen_unit_tests.cpp
  linear_algebra_unit_tests.cpp
  circuit_unit_tests.cpp
  )
target_link_libraries(unit_tests
    PUBLIC
    lgr_library)
bob_link_dependency(unit_tests PUBLIC GTest)
add_test(NAME run_unit_tests COMMAND
  "${CMAKE_CURRENT_BINARY_DIR}/unit_tests")

bob_end_subdir()
