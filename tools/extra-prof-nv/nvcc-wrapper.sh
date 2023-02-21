#!/bin/bash
COMPILER=nvcc
MSGPACK_VERSION=5.0.0
if test "${EXTRA_PROF_WRAPPER}" = off || test "${EXTRA_PROF_WRAPPER}" = OFF; then
    exec $COMPILER "$@"
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

instrumentation_arguments=("-Xcompiler" "-finstrument-functions" "-Xcompiler" "-no-pie" "-Xlinker" "-no-pie")
extra_prof_arguments=("-c" "-O2" "$compiler_dir" "-std" "c++17" "-I$(dirname "${BASH_SOURCE[0]}")/msgpack/include")
if test "${EXTRA_PROF_EVENT_TRACE}" = on || test "${EXTRA_PROF_EVENT_TRACE}" = ON; then
    extra_prof_arguments+=("-DEXTRA_PROF_EVENT_TRACE=1")
fi

if $compile_only; then
    combined=("${instrumentation_arguments[@]}" "${arguments[@]}")
    echo "EXTRA PROF COMPILE: " $COMPILER ${combined[*]}
    exec $COMPILER ${combined[*]}
else
    if [ -d "$(dirname "${BASH_SOURCE[0]}")/msgpack" ]; then
        echo "EXTRA PROF: Found msgpack"
    else
        echo "EXTRA PROF: Downloading msgpack..."
        wget -nv -O"$(dirname "${BASH_SOURCE[0]}")/msgpack-cxx.tar.gz" "https://github.com/msgpack/msgpack-c/releases/download/cpp-$MSGPACK_VERSION/msgpack-cxx-$MSGPACK_VERSION.tar.gz"
        [ $? -eq 0 ] || exit $?
        echo "EXTRA PROF: Unpacking msgpack..."
        tar -xzf "$(dirname "${BASH_SOURCE[0]}")/msgpack-cxx.tar.gz" -C "$(dirname "${BASH_SOURCE[0]}")"
        [ $? -eq 0 ] || exit $?
        mv "$(dirname "${BASH_SOURCE[0]}")/msgpack-cxx-$MSGPACK_VERSION" "$(dirname "${BASH_SOURCE[0]}")/msgpack"
        [ $? -eq 0 ] || exit $?
        echo "EXTRA PROF: Finished unpacking msgpack. Continuing..."
    fi

    # no exec otherwise this script will end here
    combined=("${extra_prof_arguments[@]}" "-o" "extra_prof_instrumentation.o" "$(dirname "${BASH_SOURCE[0]}")/extra_prof/instrumentation.cpp")
    echo "EXTRA PROF COMPILE INSTRUMENTATION: " $COMPILER ${combined[*]}
    $COMPILER ${combined[*]}
    [ $? -eq 0 ] || exit $?

    combined=("${instrumentation_arguments[@]}" "-lcupti" "-L" "$CUDA_HOME/extras/CUPTI/lib64" "extra_prof_instrumentation.o" "${arguments[@]}")
    echo "EXTRA PROF CALL: " $COMPILER ${combined[*]}
    exec $COMPILER ${combined[*]}
fi
