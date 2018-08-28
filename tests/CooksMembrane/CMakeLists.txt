get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MY_PROBLEM "CooksMembrane")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/${MY_PROBLEM}.yaml COPYONLY)
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}.osh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr ${ALL_THREAD_ARGS}
  --output-viz=${MY_PROBLEM} --input-config=${MY_PROBLEM}.yaml)
add_test(NAME ${testName} COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} END)
