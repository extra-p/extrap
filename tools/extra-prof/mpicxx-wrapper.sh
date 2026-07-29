#!/bin/bash
export EXTRA_PROF_COMPILER=mpicxx
exec bash "$(dirname "${BASH_SOURCE[0]}")/extra-prof-wrapper.sh" "$@"