get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr --kokkos-threads=1
     --output-viz=cubeMieGruniesen --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeMieGruniesen.yaml)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} -Floor 1e-10 ${CMAKE_CURRENT_SOURCE_DIR}/cubeMieGruniesen_2000_gold
       cubeMieGruniesen/steps/step_2000)
add_test(NAME ${testName} COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)
