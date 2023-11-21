#include "gpu_energy.h"
#include "../globals.h"
#include "../library/lib_extra_prof.h"
#include "commons.h"

namespace extra_prof::gpu {

void* EnergySampler::samplingThreadFunc(void* ptr) {
    extra_prof_scope sc;
    EnergySampler* self = reinterpret_cast<EnergySampler*>(ptr);
    std::deque<EnergySample>& energy_samples = self->energy_samples;
    std::deque<IdleRange>& idle_ranges = self->idle_ranges;
    std::atomic<bool>& sampling = self->sampling;

    char pciBusId[16];
    GPU_CALL(cudaDeviceGetPCIBusId(pciBusId, 16, 0));
    nvmlDevice_t device;
    NVML_CALL(nvmlDeviceGetHandleByPciBusId(pciBusId, &device));

    int res;

    energy_uj energy = 0;
    unsigned long long lastSeenTimeStamp = 0; // microseconds
    unsigned long long utilLastSeenTimeStamp = 0;

    energy_samples.emplace_back(lastSeenTimeStamp * 1000, energy);
    idle_ranges.emplace_back(utilLastSeenTimeStamp, utilLastSeenTimeStamp);

    nvmlValueType_t sampleValType;

    unsigned int maxUtilSampleCount;
    constexpr unsigned int sampleCountOffset = 5;

    nvmlReturn_t result = nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, utilLastSeenTimeStamp,
                                               &sampleValType, &maxUtilSampleCount, NULL);
    if (result == NVML_ERROR_NOT_FOUND) {
        maxUtilSampleCount = 75;
    } else {
        NVML_CALL(result);
        if (maxUtilSampleCount < 75) {
            maxUtilSampleCount = 75;
        }
    }
    assert(sampleValType == NVML_VALUE_TYPE_UNSIGNED_INT);

    maxUtilSampleCount += sampleCountOffset;
    nvmlSample_t utilSamples[maxUtilSampleCount];

    auto num_util_samples =
        self->loadUtilizationSamples(device, maxUtilSampleCount, utilSamples, utilLastSeenTimeStamp);

    unsigned long long util_pause_time_in_us =
        (utilSamples[num_util_samples - 1].timeStamp - utilSamples[0].timeStamp) / num_util_samples *
        (maxUtilSampleCount - sampleCountOffset);

    unsigned int maxSampleCount;
    result = nvmlDeviceGetSamples(device, NVML_TOTAL_POWER_SAMPLES, lastSeenTimeStamp, &sampleValType, &maxSampleCount,
                                  NULL);
    if (result == NVML_ERROR_NOT_FOUND) {
        maxSampleCount = 120;
    } else {
        NVML_CALL(result);
        if (maxSampleCount < 120) {
            maxSampleCount = 120;
        }
    }
    assert(sampleValType == NVML_VALUE_TYPE_UNSIGNED_INT);

    maxSampleCount += sampleCountOffset;
    nvmlSample_t samples[maxSampleCount];

    auto num_samples = self->loadEnergySamples(device, maxSampleCount, samples, lastSeenTimeStamp, energy);

    unsigned long long pause_time_in_us = (samples[num_samples - 1].timeStamp - samples[0].timeStamp) / num_samples *
                                          (maxSampleCount - sampleCountOffset);

    pause_time_in_us = std::min(pause_time_in_us, util_pause_time_in_us) / 2;
    struct timespec ts {
        (time_t)(pause_time_in_us / 1000000), (time_t)((pause_time_in_us % 1000000) * 1000)
    };

    self->byteSizeSamplingBuffers.store(sizeof(utilSamples) + sizeof(samples), std::memory_order_relaxed);

    while (true) {
        auto num_samples = self->loadEnergySamples(device, maxSampleCount, samples, lastSeenTimeStamp, energy);
        auto util_num_samples =
            self->loadUtilizationSamples(device, maxUtilSampleCount, utilSamples, utilLastSeenTimeStamp);
        if (num_samples == 0 && util_num_samples == 0) {
            if (!sampling.load(std::memory_order_relaxed)) {
                break;
            }
            self->processEntries(lastSeenTimeStamp, utilLastSeenTimeStamp);
            nanosleep(&ts, NULL);
        } else {
            NVML_CALL(result);
            auto start_idx = energy_samples.size();
            if (!sampling.load(std::memory_order_relaxed)) {
                break;
            }
            if (num_samples + sampleCountOffset < maxSampleCount) {
                self->processEntries(lastSeenTimeStamp, utilLastSeenTimeStamp);
                nanosleep(&ts, NULL);
            }
        }
    }
    nanosleep(&ts, NULL);
    self->loadEnergySamples(device, maxSampleCount, samples, lastSeenTimeStamp, energy);
    self->loadUtilizationSamples(device, maxUtilSampleCount, utilSamples, utilLastSeenTimeStamp);
    self->finalizeProcessingEntries();
    return 0;
}

energy_uj readEnergyFile(int fileDescriptor, char* buffer, int buffer_size) {
    ssize_t read_count = pread(fileDescriptor, buffer, buffer_size, 0);
    if (read_count == -1) {
        int errsv = errno;
        const char* error = strerror(errsv);
        throw std::runtime_error(containers::string("EXTRA PROF: ERROR: Reading energy counter: ") + error);
    }
    buffer[read_count] = 0;
    auto value = strtoull(buffer, nullptr, 10);
    if (value == 0) {
        throw std::runtime_error("EXTRA PROF: ERROR: Energy counter could not be interpreted.");
    }

    return value;
}

void* EnergySampler::fileBasedSamplingThreadFunc(void* ptr) {
    extra_prof_scope sc;
    EnergySampler* self = reinterpret_cast<EnergySampler*>(ptr);
    std::deque<EnergySample>& energy_samples = self->energy_samples;
    std::deque<IdleRange>& idle_ranges = self->idle_ranges;
    std::atomic<bool>& sampling = self->sampling;

    char pciBusId[16];
    GPU_CALL(cudaDeviceGetPCIBusId(pciBusId, 16, 0));
    nvmlDevice_t device;
    NVML_CALL(nvmlDeviceGetHandleByPciBusId(pciBusId, &device));

    int res;

    energy_uj prevEnergy = 0;
    time_point lastSeenTimeStamp = 0;
    unsigned long long utilLastSeenTimeStamp = 0;

    energy_samples.emplace_back(lastSeenTimeStamp, 0);
    idle_ranges.emplace_back(utilLastSeenTimeStamp, utilLastSeenTimeStamp);

    nvmlValueType_t sampleValType;

    unsigned int maxUtilSampleCount;
    constexpr unsigned int sampleCountOffset = 5;

    nvmlReturn_t result = nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, utilLastSeenTimeStamp,
                                               &sampleValType, &maxUtilSampleCount, NULL);
    if (result == NVML_ERROR_NOT_FOUND) {
        maxUtilSampleCount = 75;
    } else {
        NVML_CALL(result);
        if (maxUtilSampleCount < 75) {
            maxUtilSampleCount = 75;
        }
    }
    assert(sampleValType == NVML_VALUE_TYPE_UNSIGNED_INT);

    maxUtilSampleCount += sampleCountOffset;
    nvmlSample_t utilSamples[maxUtilSampleCount];

    auto num_util_samples =
        self->loadUtilizationSamples(device, maxUtilSampleCount, utilSamples, utilLastSeenTimeStamp);

    unsigned long long util_pause_time_in_us =
        (utilSamples[num_util_samples - 1].timeStamp - utilSamples[0].timeStamp) / num_util_samples *
        (maxUtilSampleCount - sampleCountOffset) / 2;

    // struct timespec ts {
    //     (time_t)(pause_time_in_us / 1000000), (time_t)((pause_time_in_us % 1000000) * 1000)
    // };

    self->byteSizeSamplingBuffers.store(sizeof(utilSamples), std::memory_order_relaxed);
    const char* EXTRA_PROF_GPU_ENERGY_COUNTER_PATH = getenv("EXTRA_PROF_GPU_ENERGY_COUNTER_PATH");
    assert(EXTRA_PROF_GPU_ENERGY_COUNTER_PATH != nullptr);

    std::cerr << "EXTRA PROF: GPU ENERGY COUNTER PATH: " << EXTRA_PROF_GPU_ENERGY_COUNTER_PATH << '\n';
    int fileDescriptor = open(EXTRA_PROF_GPU_ENERGY_COUNTER_PATH, O_RDONLY | O_CLOEXEC);

    if (fileDescriptor == -1) {
        int errsv = errno;
        const char* error = strerror(errsv);
        throw std::runtime_error(containers::string("EXTRA PROF: ERROR: Could not open: ") +
                                 EXTRA_PROF_GPU_ENERGY_COUNTER_PATH + " " + error);
    }

    char buffer[48];

    time_point pause_time = 0;

    lastSeenTimeStamp = get_timestamp();
    while (sampling.load(std::memory_order_relaxed)) {
        auto now = get_timestamp();

        auto value = readEnergyFile(fileDescriptor, buffer, sizeof(buffer));

        if (prevEnergy == value) {
            energy_samples.back().tp = now;
        } else {
            energy_samples.emplace_back(now, value);
        }

        time_point utilNextSeeTimeStamp_ns = utilLastSeenTimeStamp + util_pause_time_in_us;
        utilNextSeeTimeStamp_ns *= 1000;

        if (utilNextSeeTimeStamp_ns <= now) {
            auto util_num_samples =
                self->loadUtilizationSamples(device, maxUtilSampleCount, utilSamples, utilLastSeenTimeStamp);
        }

        auto* front = self->entry_tasks.peek();
        if (front != nullptr && front->end <= now && front->end <= utilLastSeenTimeStamp * 1000) {
            self->processEntry(front);
        }

        struct timespec ts {
            (time_t)(0), (time_t)(pause_time >> 4)
        };

        nanosleep(&ts, nullptr);
        if (prevEnergy != value) {
            pause_time = (pause_time * 31 + now - lastSeenTimeStamp) >> 5;

            lastSeenTimeStamp = now;
            prevEnergy = value;
        }
    }
    struct timespec ts {
        (time_t)(0), (time_t)(pause_time >> 4)
    };
    nanosleep(&ts, nullptr);
    auto now = get_timestamp();
    auto value = readEnergyFile(fileDescriptor, buffer, sizeof(buffer));
    if (prevEnergy == value) {
        energy_samples.back().tp = now;
    } else {
        energy_samples.emplace_back(now, value);
    }
    self->loadUtilizationSamples(device, maxUtilSampleCount, utilSamples, utilLastSeenTimeStamp);
    self->finalizeProcessingEntries();
    close(fileDescriptor);
    return 0;
}

int EnergySampler::loadEnergySamples(nvmlDevice_t device, unsigned int maxSampleCount, nvmlSample_t* samples,
                                     unsigned long long& lastSeenTimeStamp, energy_uj& energy) {
    nvmlValueType_t sampleValType;
    unsigned int sampleCount = maxSampleCount;
    nvmlReturn_t result = nvmlDeviceGetSamples(device, NVML_TOTAL_POWER_SAMPLES, lastSeenTimeStamp, &sampleValType,
                                               &sampleCount, samples);
    if (result == NVML_ERROR_NOT_FOUND) {
        return 0;
    } else {
        NVML_CALL(result);
        for (size_t i = 0; i < sampleCount; i++) {
            // (us * mW) / 1000 = uJ
            energy += ((samples[i].timeStamp - lastSeenTimeStamp) * samples[i].sampleValue.uiVal) / 1000;
            energy_samples.emplace_back(samples[i].timeStamp * 1000, energy);
            lastSeenTimeStamp = samples[i].timeStamp;
        }
    }
    return sampleCount;
}
int EnergySampler::loadUtilizationSamples(nvmlDevice_t device, unsigned int maxSampleCount, nvmlSample_t* samples,
                                          unsigned long long& lastSeenTimeStamp) {
    nvmlValueType_t sampleValType;
    unsigned int sampleCount = maxSampleCount;
    nvmlReturn_t result = nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, lastSeenTimeStamp, &sampleValType,
                                               &sampleCount, samples);
    if (result == NVML_ERROR_NOT_FOUND) {
        return 0;
    } else {
        NVML_CALL(result);
        for (size_t i = 0; i < sampleCount; i++) {
            auto& last_entry = idle_ranges.back();
            if (samples[i].sampleValue.uiVal == 0) {
                last_entry.end = samples[i].timeStamp / 1000;
            } else if (last_entry.end > 0) {
                idle_ranges.emplace_back(samples[i].timeStamp / 1000, 0);
            } else {
                last_entry.begin = samples[i].timeStamp / 1000;
            }
            lastSeenTimeStamp = samples[i].timeStamp;
        }
    }
    return sampleCount;
}

void EnergySampler::processEntries(unsigned long long lastSeen_us, unsigned long long utilLastSeen_us) {

    auto* front = entry_tasks.peek();

    time_point lastSeen_ns = lastSeen_us * 1000;
    time_point utilLastSeen_ns = utilLastSeen_us * 1000;

    while (front != nullptr && front->end <= lastSeen_ns && front->end <= utilLastSeen_ns) {
        processEntry(front);
        // std::cout << "Processed: " << front->node << ' ' << front->node->name() << " " << processEntry(front) <<
        // "\n";

        front = entry_tasks.peek();
    }
}

void EnergySampler::finalizeProcessingEntries() {

    auto* front = entry_tasks.peek();
    while (front != nullptr) {
        processEntry(front);
        // std::cout << "Processed: " << front->node << ' ' << front->node->name() << " " << processEntry(front) <<
        // "\n";

        front = entry_tasks.peek();
    }
}

energy_uj EnergySampler::processEntry(EntryTask* front) {
    auto idle_begin = std::upper_bound(idle_ranges.begin(), idle_ranges.end(), front->begin,
                                       [](const time_point& a, const IdleRange& b) { return a < b.end; });

    auto idle_end = std::upper_bound(idle_ranges.begin(), idle_ranges.end(), front->end,
                                     [](const time_point& a, const IdleRange& b) { return a < b.end; });
    energy_uj energy = 0;

    if (idle_begin == idle_end) {
        if (idle_begin->begin < front->end) {
            // function end lies within idle range
            auto begin = std::max(front->begin, idle_begin->begin);
            energy += getEnergy(begin, front->end);
        } else {
            // function start and end before idle range
            // nothing to do
        }
    } else {
        auto iterator = idle_begin;
        if (iterator->begin < front->begin) { // front->begin < iterator->end always satisfied
            // if function range starts within first idle range
            energy += getEnergy(front->begin, iterator->end);
            ++iterator;
        }
        for (; iterator != idle_end; ++iterator) {
            energy += getEnergy(iterator->begin, iterator->end);
        }
        if (idle_end->begin <= front->end) { // front->end < idle_end->end always satisfied
            // if function range ends within last idle range
            energy += getEnergy(idle_end->begin, front->end);
        }
    }
    front->node->energy_gpu += energy;
    entry_tasks.pop();
    return front->node->energy_gpu;
}

void EnergySampler::start() {
#ifdef EXTRA_PROF_ENERGY
    NVML_CALL(nvmlInit_v2());
    sampling = true;

    const char* EXTRA_PROF_GPU_ENERGY_COUNTER_PATH = getenv("EXTRA_PROF_GPU_ENERGY_COUNTER_PATH");
    if (EXTRA_PROF_GPU_ENERGY_COUNTER_PATH != nullptr) {
        create_pthread_without_instrumentation(&samplingThread, NULL, &fileBasedSamplingThreadFunc, this);
    } else {
        create_pthread_without_instrumentation(&samplingThread, NULL, &samplingThreadFunc, this);
    }
    pthread_setname_np(samplingThread, "EP_E_Sampler");
#endif
}

void EnergySampler::stop() {
#ifdef EXTRA_PROF_ENERGY
    sampling = false;
    void* thread_return;
    pthread_join(samplingThread, &thread_return);
#endif
}
energy_uj EnergySampler::getEnergy(time_point start_tp, time_point end_tp) {
#ifdef EXTRA_PROF_ENERGY

    auto startIt = std::lower_bound(energy_samples.cbegin(), energy_samples.cend(), start_tp);
    auto endIt = std::upper_bound(energy_samples.cbegin(), energy_samples.cend(), end_tp);
    auto beforeStartIt = startIt - 1;
    auto beforeEndIt = endIt - 1;

    if (startIt == endIt) {
        // start_tp and end_tp lie within one sample
        auto delta_start = end_tp - start_tp;
        auto delta_sampling = startIt->tp - beforeStartIt->tp;
        auto delta_energy = startIt->energy - beforeStartIt->energy;
        return delta_energy * delta_start / delta_sampling;
    }

    auto delta_start = startIt->tp - start_tp;
    auto delta_sampling = startIt->tp - beforeStartIt->tp;
    auto delta_energy = startIt->energy - beforeStartIt->energy;

    auto start_energy = delta_energy * delta_start / delta_sampling;

    auto middle_energy = beforeEndIt->energy - startIt->energy;

    auto delta_end = end_tp - beforeEndIt->tp;
    auto delta_end_sampling = endIt->tp - beforeEndIt->tp;
    auto delta_end_energy = endIt->energy - beforeEndIt->energy;

    auto end_energy = delta_end_energy * delta_end / delta_end_sampling;

    return start_energy + middle_energy + end_energy;
#else
    return 0;
#endif
}

void EnergySampler::addEntryTask(CallTreeNode* node, time_point begin, time_point end) {
#ifdef EXTRA_PROF_ENERGY
    if (!sampling) {
        return;
    }
    extra_prof_scope sc;
    assert(GLOBALS.main_thread == pthread_self());
    entry_tasks.emplace(node, begin, end);
#endif
}

std::deque<EnergySample>& EnergySampler::getSamples() {
    if (sampling) {
        throw std::logic_error("You cannot read samples while sampling");
    }
    return energy_samples;
}

} // namespace extra_prof::gpu