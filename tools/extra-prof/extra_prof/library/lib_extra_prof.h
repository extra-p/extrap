#include "../common_types.h"

namespace extra_prof {
EXTRA_PROF_SO_EXPORT int create_pthread_without_instrumentation(pthread_t* thread, const pthread_attr_t* attr,
                                                                void* (*start_routine)(void*), void* arg);
}