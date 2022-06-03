#!/bin/bash
if test "${EXTRA_PROF_WRAPPER}" = off || test "${EXTRA_PROF_WRAPPER}" = OFF; then
    exec nvcc "$@"
fi
arguments=("$@")
compile_only=false
while test $# -gt 0; do
    #echo "$#: $1"
    case "$1" in
    -c)
        compile_only=true
        ;;
    --compile)
        compile_only=true
        ;;
    -ccbin)
        compiler_dir="$1 $2"
        ;;
    --compiler-bindir=*)
        compiler_dir=$1
        ;;
    esac
    shift
done
instrumentation_arguments="-Xcompiler -finstrument-functions"
if $compile_only; then
    #    echo "Compile only: ${arguments[@]}"
    exec nvcc $instrumentation_arguments "${arguments[@]}"
else
    # no exec otherwise this script will end here
    nvcc -c -g $compiler_dir -std c++17 -I${CUDA_PATH}/include/nvtx3 $(dirname "${BASH_SOURCE[0]}")/instrumentation.cpp -o extra_profiler.o
    [ $? -eq 0 ] || exit $?
    exec nvcc -L ${CUDA_PATH}/lib64/libnvToolsExt.so $instrumentation_arguments extra_profiler.o "${arguments[@]}"
fi
