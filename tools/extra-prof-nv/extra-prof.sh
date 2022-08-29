#!/bin/bash
if [ -z ${EXTRA_PROF_EXPERIMENT_DIRECTORY+x} ]; then
    EXTRA_PROF_EXPERIMENT_DIRECTORY=$EXTRA_PROF_EXPERIMENT_DIRECTORY
elif [ -z ${SCOREP_EXPERIMENT_DIRECTORY+x} ]; then
    EXTRA_PROF_EXPERIMENT_DIRECTORY=$SCOREP_EXPERIMENT_DIRECTORY
else
    EXTRA_PROF_EXPERIMENT_DIRECTORY="extra_prof_result$(date)"
fi
nsys profile -o "$EXTRA_PROF_EXPERIMENT_DIRECTORY/profile%q{SLURM_PROCID}" --export sqlite -b dwarf --cudabacktrace none -s none -t cuda,nvtx -x true -f true "$@"
