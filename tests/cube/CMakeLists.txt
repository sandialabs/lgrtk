# copy mesh files from current source dir to binary dir
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cube.dmg
               ${CMAKE_CURRENT_BINARY_DIR}/cube.dmg COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/cube.osh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr --kokkos-threads=1
         --output-viz=cube --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} ${CMAKE_CURRENT_SOURCE_DIR}/cube_1000_gold
      cube/steps/step_1000)
add_test(NAME ${testName} COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)

build_mpi_test_string(MPI_TEST 2 ${LGR_BINARY_DIR}/lgr
         --kokkos-threads=1 --output-viz=cube_mpi --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cube.yaml)
build_mpi_test_string(DIFF_TEST 2 ${VTKDIFF} ${CMAKE_CURRENT_SOURCE_DIR}/cube_1000_gold
      cube_mpi/steps/step_1000)
add_test(NAME ${testName}_MPI COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr  --kokkos-threads=1
         --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeRestart1.yaml --output-viz=cubeRestart)
add_test(NAME cubeRestart1 COMMAND ${MPI_TEST})

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr  --kokkos-threads=1
         --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeRestart2.yaml --output-viz=cubeRestart)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} ${CMAKE_CURRENT_SOURCE_DIR}/cube_1000_gold
      cubeRestart/steps/step_1000)
add_test(NAME cubeRestart2 COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)

build_mpi_test_string(MPI_TEST 2 ${LGR_BINARY_DIR}/lgr
         --kokkos-threads=1 --output-viz=cubeRestart2_MPI --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeRestart2.yaml)
build_mpi_test_string(DIFF_TEST 2 ${VTKDIFF} ${CMAKE_CURRENT_SOURCE_DIR}/cube_1000_gold
      cubeRestart2_MPI/steps/step_1000)
add_test(NAME cubeRestart2_MPI COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)
