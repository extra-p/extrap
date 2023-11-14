#include "../calltree_node.cpp"
#include "energy.cpp"
#ifdef EXTRA_PROF_GPU
#ifdef EXTRA_PROF_ENERGY
#include "gpu_energy.cpp"
#endif
#include "gpu_instrumentation.cpp"
#endif
#include "function_instrumentation.cpp"

// #include "globals.cpp"