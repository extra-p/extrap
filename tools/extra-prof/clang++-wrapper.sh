#!/bin/bash
export EXTRA_PROF_COMPILER=clang++
export EXTRA_PROF_ADVANCED_INSTRUMENTATION=off
exec bash "$(dirname "${BASH_SOURCE[0]}")/extra-prof-wrapper.sh" "$@"