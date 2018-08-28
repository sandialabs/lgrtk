get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${testName}_adapt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/${testName}_adapt.yaml COPYONLY)
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/${testName}.osh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr ${ALL_THREAD_ARGS}
     --output-viz=${testName}_adapt --input-config=${testName}_adapt.yaml)
SET(DIFF_TEST "")
add_test(NAME ${testName}_adapt COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)
