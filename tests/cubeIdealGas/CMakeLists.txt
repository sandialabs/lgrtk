get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr --kokkos-threads=1
     --output-viz=cubeIdealGas --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeIdealGas.yaml)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} -Floor 1e-10 ${CMAKE_CURRENT_SOURCE_DIR}/cubeIdealGas_2000_gold
       cubeIdealGas/steps/step_2000)
add_test(NAME ${testName} COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
if (HOSTNAME MATCHES "^cee")
  set (GOLD_COPY cubeIdealGas_adapt_2000_cee_gold)
  set (TOL 5e-13)
elseif (HOSTNAME MATCHES "^ellis")
# KNL test
  set (GOLD_COPY cubeIdealGas_adapt_2000_ellis_gold)
  set (TOL 5e-13)
elseif (HOSTNAME MATCHES "^ascicgpu")
# NVIDIA Kepler test
  set (GOLD_COPY cubeIdealGas_adapt_2000_cuda_gold)
  set (TOL 5e-11)
elseif (HOSTNAME MATCHES "^white")
# NVIDIA Pascal test
  set (GOLD_COPY cubeIdealGas_adapt_2000_cuda_gold)
  set (TOL 5e-11)
elseif (HOSTNAME MATCHES "^ride")
# NVIDIA Pascal test
  set (GOLD_COPY cubeIdealGas_adapt_2000_cuda_gold)
  set (TOL 5e-11)
endif ()
build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr --kokkos-threads=1
     --output-viz=cubeIdealGas_adapt --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeIdealGas_adapt.yaml)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} -Floor ${TOL} ${CMAKE_CURRENT_SOURCE_DIR}/${GOLD_COPY}
       cubeIdealGas_adapt/steps/step_2000)
add_test(NAME ${testName}_adapt COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)
if (GOLD_COPY)
  build_mpi_test_string(MPI_TEST 2 ${LGR_BINARY_DIR}/lgr --kokkos-threads=1 --kokkos-ndevices=2
       --output-viz=cubeIdealGas_adapt_mpi --input-config=${CMAKE_CURRENT_SOURCE_DIR}/cubeIdealGas_adapt.yaml)
  build_mpi_test_string(DIFF_TEST 2 ${VTKDIFF} -Floor ${TOL} ${CMAKE_CURRENT_SOURCE_DIR}/${GOLD_COPY}
         cubeIdealGas_adapt_mpi/steps/step_2000)
  add_test(NAME ${testName}_adapt_mpi COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)
endif()
