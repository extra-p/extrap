#!/bin/bash
export EXTRA_PROF_COMPILER=gcc
exec bash "$(dirname "${BASH_SOURCE[0]}")/extra-prof-wrapper.sh" "$@"