get_filename_component(testName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(MY_PROBLEM "Noh")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/assoc.txt
               ${CMAKE_CURRENT_BINARY_DIR}/assoc.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}_adapt.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/${MY_PROBLEM}_adapt.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/${MY_PROBLEM}.yaml COPYONLY)
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}_gold
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}.osh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
if (HOSTNAME MATCHES "^cee")
  set (GOLD_COPY _adapt_cee_gold)
elseif (HOSTNAME MATCHES "^ellis")
# KNL
  set (GOLD_COPY _adapt_ellis_gold)
elseif (HOSTNAME MATCHES "^ascicgpu")
# NVIDIA Kepler
  set (GOLD_COPY _adapt_cuda_gold)
elseif (HOSTNAME MATCHES "^white")
# NVIDIA Pascal
  set (GOLD_COPY _adapt_cuda_gold)
elseif (HOSTNAME MATCHES "^ride")
# NVIDIA Pascal
  set (GOLD_COPY _adapt_cuda_gold)
else ()
  set (GOLD_COPY _adapt_cee_gold)
endif ()

# file(COPY        ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}${GOLD_COPY}
#      DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr ${ALL_THREAD_ARGS}
  --output-viz=${MY_PROBLEM} --input-config=${MY_PROBLEM}.yaml)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} -Floor 1e-10 ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}_gold ${MY_PROBLEM}/steps/step_100)
add_test(NAME ${testName} COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)

build_mpi_test_string(MPI_TEST 1 ${LGR_BINARY_DIR}/lgr ${ALL_THREAD_ARGS}
     --output-viz=${MY_PROBLEM}_adapt --input-config=${MY_PROBLEM}_adapt.yaml)
build_mpi_test_string(DIFF_TEST 1 ${VTKDIFF} -Floor 1e-10 ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}${GOLD_COPY} ${MY_PROBLEM}_adapt/steps/step_50)
add_test(NAME ${testName}_adapt COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)

build_mpi_test_string(MPI_TEST 2 ${LGR_BINARY_DIR}/lgr ${ALL_THREAD_ARGS}
     --output-viz=${MY_PROBLEM}_adapt_mpi --input-config=${MY_PROBLEM}_adapt.yaml)
build_mpi_test_string(DIFF_TEST 2 ${VTKDIFF} -Floor 1e-10 ${CMAKE_CURRENT_SOURCE_DIR}/${MY_PROBLEM}${GOLD_COPY} ${MY_PROBLEM}_adapt_mpi/steps/step_50)
add_test(NAME ${testName}_adapt_mpi COMMAND ${CMAKE_SOURCE_DIR}/tests/runtest.sh FIRST ${MPI_TEST} SECOND ${DIFF_TEST} END)
