
#pragma once
#include <memory>
typedef uint64_t time_point;
typedef uint64_t duration;

template <typename T>
class OnExit {
public:
    OnExit(T t) : t(t) {}
    ~OnExit() { t(); }
    T t;
};