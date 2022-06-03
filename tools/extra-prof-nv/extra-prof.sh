#!/bin/bash
nsys profile -o "$SCOREP_EXPERIMENT_DIRECTORY/profile%q{SLURM_PROCID}" --export sqlite -b dwarf --cudabacktrace none -s none -t cuda,nvtx -x true -f true $@