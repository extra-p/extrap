#include "address_mapping.cpp"
#include "exit_wrapper.cpp"
#include "globals.cpp"

#if __has_include(<omp-tools.h>)
#include "ompt_wrapper.cpp"
#elif (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#include "gomp_wrapper.cpp"
#else
#warning "EXTRA PROF: WARNING: Could not apply any instrumentation for OpenMP. " \
    "Using OpenMP will work, but all measurements will be attributed to the first OpenMP region."
#endif

#include "pthread_wrapper.cpp"