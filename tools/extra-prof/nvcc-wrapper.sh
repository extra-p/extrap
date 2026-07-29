#!/bin/bash
export EXTRA_PROF_COMPILER=nvcc
export EXTRA_PROF_COMPILER_OPTION_REDIRECT="-Xcompiler"
exec bash "$(dirname "${BASH_SOURCE[0]}")/extra-prof-wrapper.sh" "$@"