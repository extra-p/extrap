#pragma once
#include "../common_types.h"

#include <iostream>
#include <unistd.h>

namespace extra_prof::cpu::energy {
#ifdef EXTRA_PROF_ENERGY

class EnergyMeasurementSystem {
private:
    std::atomic<energy_uj> energy = 0;
    pthread_t samplingThread;
    std::atomic<bool> sampling = true;
    std::vector<EnergyCounter> energyCounters;

    static void* samplingThreadFunc(void* ptr);

public:
    void start();

    void stop() {
        sampling = false;
        void* thread_return;
        pthread_join(samplingThread, &thread_return);
        for (auto& energyCounter : energyCounters) {
            close(energyCounter.fileDescriptor);
        }
        energyCounters.clear();
    }

    energy_uj getEnergy() { return energy.load(std::memory_order_relaxed); }
};

#endif
} // namespace extra_prof::cpu::energy
