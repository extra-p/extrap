
#pragma once

#include <memory>
typedef uint64_t time_point;
typedef uint64_t duration;
typedef unsigned long long energy_uj;

template <typename T>
class OnExit {
public:
    OnExit(T t) : t(t) {}
    ~OnExit() { t(); }
    T t;
};

#ifdef __GNUC__
#define EP_INLINE __attribute__((always_inline)) inline
#else
#define EP_INLINE inline
#endif

#define EXTRA_PROF_SO_EXPORT