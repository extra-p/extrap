
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

namespace extra_prof {

typedef uint32_t ScorepRegion;

enum RegionType : uint8_t {
    FUNCTION_PTR_REGION = 0,
    NAMED_REGION = 1,
    SCOREP_REGION = 2,
#ifdef EXTRA_PROF_GPU
    CUDA_REGION = 3,
    REGIONTYPES_LENGTH = 4,
#else
    CUDA_REGION = 255,
    REGIONTYPES_LENGTH = 3,
#endif
    UNDEFINED_REGION = 255
};

union RegionID {
    void* function_ptr;
    const char* name;
    ScorepRegion scorep_region;
    uint64_t comparison_value;
};

static RegionID temp_region;
static_assert(sizeof(temp_region.comparison_value) == sizeof(RegionID));

EP_INLINE RegionID toRegionID(const char* name) {
    RegionID id;
    id.comparison_value = 0;
    id.name = name;
    return id;
}

EP_INLINE RegionID toRegionID(void* function_ptr) {
    RegionID id;
    id.comparison_value = 0;
    id.function_ptr = function_ptr;
    return id;
}
EP_INLINE RegionID toRegionID(uint32_t scorep_region) {
    RegionID id;
    id.comparison_value = 0;
    id.scorep_region = scorep_region;
    return id;
}

EP_INLINE std::size_t hash_value(const RegionID& region) noexcept { return region.comparison_value; }
EP_INLINE bool operator==(const RegionID& a, const RegionID& b) { return a.comparison_value == b.comparison_value; }

struct EnergyCounter {
    int fileDescriptor = 0;
    unsigned long long maxValue = 0;
    unsigned long long previousValue = 0;
    unsigned long long totalValue = 0;

    EnergyCounter() {}
    EnergyCounter(int fd, unsigned long long maxValue_) : fileDescriptor(fd), maxValue(maxValue_) {}
};
} // namespace extra_prof

template <>
struct std::hash<extra_prof::RegionID> {
    EP_INLINE std::size_t operator()(const extra_prof::RegionID& region) const noexcept {
        return region.comparison_value;
    }
};
