#!/bin/bash

while [ $# -gt 0 ]
do
  case $1 in
  #Empty the command string
  FIRST)
    command_args=""
    ;;
  #Execute the first command, and empty the command string
  SECOND|THIRD)
    $command_args
    OUT=$?
    if [ $OUT -ne 0 ]; then
      echo Command: $command_args - exited with an error
      exit $OUT
    fi
    echo Command: $command_args - passed test.
    command_args=""
    ;;
  #Execute the last command
  END)
    $command_args
    OUT=$?
    if [ $OUT -ne 0 ]; then
      echo Command: $command_args - exited with an error
      exit $OUT
    fi
    echo Command: $command_args - passed test.
    exit 0
    ;;
  #All other args are captured
  *)
    command_args="$command_args $1"
    ;;
  esac

  shift
done
