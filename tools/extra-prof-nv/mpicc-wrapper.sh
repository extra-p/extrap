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

instrumentation_arguments=("-finstrument-functions" "-no-pie")
extra_prof_arguments=("-c" "-O2" "-std" "c++17" "-I$(dirname "${BASH_SOURCE[0]}")/msgpack/include")
if test "${EXTRA_PROF_EVENT_TRACE}" = on || test "${EXTRA_PROF_EVENT_TRACE}" = ON; then
    extra_prof_arguments+=("-DEXTRA_PROF_EVENT_TRACE=1")
fi

if $compile_only; then
    combined=("${instrumentation_arguments[@]}" "${arguments[@]}")
    echo "EXTRA PROF CALL: " $COMPILER ${combined[*]}
    exec $COMPILER ${combined[*]}
else
    if [ -d "$(dirname "${BASH_SOURCE[0]}")/msgpack" ]; then
        echo "Found msgpack"
    else
        echo "Downloading msgpack..."
        wget -nv -O"$(dirname "${BASH_SOURCE[0]}")/msgpack-cxx.tar.gz" "https://github.com/msgpack/msgpack-c/releases/download/cpp-$MSGPACK_VERSION/msgpack-cxx-$MSGPACK_VERSION.tar.gz"
        tar -xzf -C "$(dirname "${BASH_SOURCE[0]}")" "$(dirname "${BASH_SOURCE[0]}")/msgpack-cxx.tar.gz"
        mv "$(dirname "${BASH_SOURCE[0]}")/msgpack-cxx-$MSGPACK_VERSION" "$(dirname "${BASH_SOURCE[0]}")/msgpack"
        echo "Finished unpacking msgpack. Continuing..."
    fi

    # no exec otherwise this script will end here
    combined=("${extra_prof_arguments[@]}" "-o" "extra_prof_instrumentation.o" "$(dirname "${BASH_SOURCE[0]}")/extra_prof/instrumentation.cpp")
    echo "EXTRA PROF CALL: " $COMPILER ${combined[*]}
    $COMPILER ${combined[*]}
    [ $? -eq 0 ] || exit $?

    combined=("${instrumentation_arguments[@]}" "-lcupti" "extra_prof_instrumentation.o" "${arguments[@]}")
    echo "EXTRA PROF CALL: " $COMPILER ${combined[*]}
    exec $COMPILER ${combined[*]}
fi