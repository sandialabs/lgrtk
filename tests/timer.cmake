# find 'time' binary


find_program( BINARY_SEARCH_RESULT time
              HINTS /usr/bin /bin
              NO_DEFAULT_PATH )

if( BINARY_SEARCH_RESULT MATCHES "NOTFOUND" )
  message(STATUS "INFO: 'time' executable not found.")
  message(STATUS "INFO: ${TEST_NAME} test will not be timed")
  set( TEST_COMMAND )
else()
  set( TEST_COMMAND "${BINARY_SEARCH_RESULT} -f 'cmake_test_user_time: %U'" )
endif()

## empty output file
execute_process(COMMAND bash "-c" "echo '' > cmake_timing_data")

## test
set( TEST_COMMAND "${TEST_COMMAND} ${LGR_EXECUTABLE} --output-viz=output_data --input-config=${CONFIG_FILE} &>> cmake_timing_data" )
foreach(timed_run RANGE 1 ${NUM_RUNS})
  execute_process(COMMAND bash "-c" "${TEST_COMMAND}" RESULT_VARIABLE HAD_ERROR)
  if (HAD_ERROR)
    message(FATAL_ERROR "FAILED: ${TEST_COMMAND}")
  endif()
endforeach(timed_run)

## compute average run time
execute_process(COMMAND bash "-c" "awk '/cmake_test_user_time/{print (total_time+=$2)/${NUM_RUNS}}' cmake_timing_data | tail -1 > cmake_timing_result")

file(READ cmake_timing_result TEST_RUN_TIME)

message(STATUS "TARGET_RUN_TIME: ${RUN_TIME}")
message(STATUS "CURRENT_RUN_TIME: ${TEST_RUN_TIME}")

if( ${TEST_RUN_TIME} GREATER ${RUN_TIME} )
  message(FATAL_ERROR "Test is slower!")
else()
  message(STATUS "Test time: ${TEST_RUN_TIME}-- Compare to: ${RUN_TIME}")
endif()
