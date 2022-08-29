#!/bin/bash
COMPILER=mpicc
if test "${EXTRA_PROF_WRAPPER}" = off || test "${EXTRA_PROF_WRAPPER}" = OFF; then
    exec $COMPILER "$@"
fi
arguments=("$@")
compile_only=false
while test $# -gt 0; do
    # echo "$#: $1"
    case "$1" in
    -c)
        compile_only=true
        ;;
    --compile)
        compile_only=true
        ;;
    esac
    shift
done
instrumentation_arguments="-finstrument-functions"
if $compile_only; then
    # echo "Compile only: $arguments"
    exec $COMPILER $instrumentation_arguments "${arguments[@]}"
else
    # no exec otherwise this script will end here
    $COMPILER -c -g -I"${CUDA_PATH}/include/nvtx3" "$(dirname "${BASH_SOURCE[0]}")"/instrumentation.cpp -o extra_profiler.o
    [ $? -eq 0 ] || exit $?
    exec $COMPILER -L "${CUDA_PATH}/lib64/libnvToolsExt.so" $instrumentation_arguments extra_profiler.o "${arguments[@]}"
fi
