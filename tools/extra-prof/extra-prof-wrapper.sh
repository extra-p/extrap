#!/bin/bash
MSGPACK_VERSION=5.0.0

if [ -z ${EXTRA_PROF_COMPILER+y} ]; then
    >&2 echo "EXTRA PROF ERROR: No compiler set, either set EXTRA_PROF_COMPILER or use one of the compiler-specific wrappers."
    exit 1
fi

if [ "${EXTRA_PROF_WRAPPER}" = off ] || [ "${EXTRA_PROF_WRAPPER}" = OFF ]; then
    exec $EXTRA_PROF_COMPILER "$@"
fi
arguments=("$@")
compile_only=false
shared_library=false
while [ $# -gt 0 ]; do
    #echo "$#: $1"
    case "$1" in
    -c)
        compile_only=true
        ;;
    --compile)
        compile_only=true
        ;;
    -shared)
        shared_library=true
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

extra_prof_root="$(dirname "${BASH_SOURCE[0]}")"
msg_pack_root="$extra_prof_root/msgpack"
instrumentation_arguments=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-finstrument-functions")
extra_prof_arguments=("$extra_prof_optimization" "-std=c++17" "-I$msg_pack_root/include" "$extra_prof_event_trace")
link_extra_prof_wrap="-Xlinker --no-as-needed -Xlinker --rpath=. -l_extra_prof -Xlinker --as-needed -L."

if $shared_library; then
    extra_prof_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fPIC")
fi

if [ $EXTRA_PROF_COMPILER = "nvcc" ]; then
    extra_prof_arguments+=("$compiler_dir")
fi

if [ "${EXTRA_PROF_GPU}" != "off" ] && [ "${EXTRA_PROF_GPU}" != "OFF" ]; then
    extra_prof_arguments+=("-DEXTRA_PROF_GPU=1")
fi

if [ "${EXTRA_PROF_EVENT_TRACE}" = on ] || [ "${EXTRA_PROF_EVENT_TRACE}" = ON ]; then
    extra_prof_arguments+=("-DEXTRA_PROF_EVENT_TRACE=1")
fi

if [ "${EXTRA_PROF_DEBUG_BUILD}" = on ] || [ "${EXTRA_PROF_DEBUG_BUILD}" = ON ]; then
    extra_prof_arguments+=("-g -O0")
else
    extra_prof_arguments+=("-O2")
fi

if [ "${EXTRA_PROF_DEBUG_SANITIZE}" = on ] || [ "${EXTRA_PROF_DEBUG_SANITIZE}" = ON ]; then
    instrumentation_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fsanitize=address")
    extra_prof_arguments+=("$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fsanitize=address")
fi

if $compile_only; then
    combined=("${instrumentation_arguments[@]}" "${arguments[@]}")
    echo "EXTRA PROF COMPILE: " $EXTRA_PROF_COMPILER ${combined[*]}
    exec $EXTRA_PROF_COMPILER ${combined[*]}
else
    if [ -d "$extra_prof_root/msgpack" ]; then
        echo "EXTRA PROF: Found msgpack"
    else
        echo "EXTRA PROF: Downloading msgpack..."
        wget -nv -O"$extra_prof_root/msgpack-cxx.tar.gz" "https://github.com/msgpack/msgpack-c/releases/download/cpp-$MSGPACK_VERSION/msgpack-cxx-$MSGPACK_VERSION.tar.gz"
        [ $? -eq 0 ] || exit $?
        echo "EXTRA PROF: Unpacking msgpack..."
        tar -xzf "$extra_prof_root/msgpack-cxx.tar.gz" -C "$extra_prof_root"
        [ $? -eq 0 ] || exit $?
        mv "$extra_prof_root/msgpack-cxx-$MSGPACK_VERSION" "$msg_pack_root"
        [ $? -eq 0 ] || exit $?
        echo "EXTRA PROF: Finished unpacking msgpack. Continuing..."
    fi

    combined=("--shared" "${extra_prof_arguments[@]}" "$EXTRA_PROF_COMPILER_OPTION_REDIRECT" "-fPIC" "-o" "lib_extra_prof.so" "$extra_prof_root/extra_prof/library/lib_extra_prof.cpp")

    echo "EXTRA PROF COMPILE LIBRARY: " $EXTRA_PROF_COMPILER ${combined[*]}
    $EXTRA_PROF_COMPILER ${combined[*]}
    [ $? -eq 0 ] || exit $?

    # no exec otherwise this script will end here
    combined=("-c" "${extra_prof_arguments[@]}" "-o" "extra_prof_instrumentation.o" "$extra_prof_root/extra_prof/instrumentation/instrumentation.cpp") 
    echo "EXTRA PROF COMPILE INSTRUMENTATION: " $EXTRA_PROF_COMPILER ${combined[*]}
    $EXTRA_PROF_COMPILER ${combined[*]}
    [ $? -eq 0 ] || exit $?


    combined=("$link_extra_prof_wrap" "${instrumentation_arguments[@]}"  "extra_prof_instrumentation.o" "${arguments[@]}")

    if [ "${EXTRA_PROF_GPU}" != "off" ] && [ "${EXTRA_PROF_GPU}" != "OFF" ]; then
        combined+=("-lcupti -lnvperf_host -lnvperf_target -L$CUDA_HOME/extras/CUPTI/lib64")
    fi

    echo "EXTRA PROF CALL: " $EXTRA_PROF_COMPILER ${combined[*]}
    exec $EXTRA_PROF_COMPILER ${combined[*]}
fi
