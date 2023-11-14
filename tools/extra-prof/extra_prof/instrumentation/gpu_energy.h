#pragma once
#include "../calltree_node.h"
#include "../containers/pair.h"
#include "../vendor/readerwriterqueue/readerwriterqueue.h"
#include <deque>
#include <mutex>
#include <nvml.h>

#define NVML_CALL(call)                                                                                                \
    do {                                                                                                               \
        nvmlReturn_t _status = call;                                                                                   \
        if (_status != NVML_SUCCESS) {                                                                                 \
            const char* errstr = nvmlErrorString(_status);                                                             \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", __FILE__, __LINE__, #call, errstr);   \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace extra_prof::gpu {
struct IdleRange {
    time_point begin;
    time_point end;
    IdleRange() {}
    IdleRange(time_point start_, time_point end_) : begin(start_), end(end_) {}
};

struct EnergySample {
    time_point tp;
    energy_uj energy;
    EnergySample() {}
    EnergySample(time_point tp_, energy_uj energy_) : tp(tp_), energy(energy_) {}

    bool operator<(const EnergySample& rhs) const { return this->tp < rhs.tp; }
    bool operator<(const time_point& rhs) const { return this->tp < rhs; }
};

inline bool operator<(const time_point& lhs, const EnergySample& rhs) { return lhs < rhs.tp; }

class EnergySampler {
private:
    struct EntryTask {
        CallTreeNode* node;
        time_point begin;
        time_point end;
        EntryTask() {}
        EntryTask(CallTreeNode* node_, time_point begin_, time_point end_) : node(node_), begin(begin_), end(end_) {}
    };

    std::atomic<size_t> byteSizeSamplingBuffers = 0;
    std::deque<EnergySample> energy_samples;
    std::deque<IdleRange> idle_ranges;
    pthread_t samplingThread;
    std::atomic<bool> sampling = true;

    moodycamel::ReaderWriterQueue<EntryTask> entry_tasks;

    static void* samplingThreadFunc(void* ptr);
    static void* fileBasedSamplingThreadFunc(void* ptr);

    void processEntries(unsigned long long lastSeen_us, unsigned long long utilLastSeen_us);
    void processEntry(EntryTask* front);

    int loadEnergySamples(nvmlDevice_t device, unsigned int maxSampleCount, nvmlSample_t* samples,
                          unsigned long long& lastSeenTimeStamp, energy_uj& energy);

    int loadUtilizationSamples(nvmlDevice_t device, unsigned int maxSampleCount, nvmlSample_t* samples,
                               unsigned long long& lastSeenTimeStamp);

public:
    void start();

    void stop();

    std::deque<EnergySample>& getSamples();

    void addEntryTask(CallTreeNode* node, time_point begin, time_point end);

    energy_uj getEnergy(time_point start_tp, time_point end_tp);

    size_t getByteSize() {
        return byteSizeSamplingBuffers.load(std::memory_order_relaxed) + energy_samples.size() * sizeof(EnergySample);
        +entry_tasks.size_approx() * sizeof(EntryTask);
    }
};

} // namespace extra_prof::gpu