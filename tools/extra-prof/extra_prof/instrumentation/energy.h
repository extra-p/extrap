#include "../common_types.h"
#include "../globals.h"
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

namespace extra_prof::cpu::energy {
#ifdef EXTRA_PROF_ENERGY
void initializeEnergy() {

    const char* EXTRA_PROF_CPU_ENERGY_COUNTER_PATH = getenv("EXTRA_PROF_CPU_ENERGY_COUNTER_PATH");
    if (EXTRA_PROF_CPU_ENERGY_COUNTER_PATH != nullptr) {
        std::cerr << "EXTRA PROF: CPU ENERGY COUNTER PATH: " << EXTRA_PROF_CPU_ENERGY_COUNTER_PATH << '\n';
        int fd = open(EXTRA_PROF_CPU_ENERGY_COUNTER_PATH, O_RDONLY | O_CLOEXEC);
        if (fd == -1) {
            int errsv = errno;
            const char* error = strerror(errsv);
            throw std::runtime_error(containers::string("EXTRA PROF: ERROR: ") + EXTRA_PROF_CPU_ENERGY_COUNTER_PATH +
                                     " " + error);
        }
        GLOBALS.energyCounters.emplace_back(fd, std::numeric_limits<energy_uj>::max());
    }

    const std::filesystem::path rapl_folder{"/sys/class/powercap/intel-rapl"};
    for (auto const& dir_entry : std::filesystem::directory_iterator{rapl_folder}) {
        containers::string dir_name = dir_entry.path().filename().string();
        if (dir_name.rfind("intel-rapl:", 0) == 0) { // pos=0 limits the search to the prefix

            auto path = dir_entry.path() / "energy_uj";
            int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);

            if (fd == -1) {
                int errsv = errno;
                const char* error = strerror(errsv);
                throw std::runtime_error(containers::string("EXTRA PROF: ERROR: ") + path.string() + " " + error);
            }

            auto max_path = dir_entry.path() / "max_energy_range_uj";
            int max_fd = open(max_path.c_str(), O_RDONLY | O_CLOEXEC);

            if (max_fd == -1) {
                int errsv = errno;
                const char* error = strerror(errsv);
                throw std::runtime_error(containers::string("EXTRA PROF: ERROR: ") +
                                         (dir_entry.path() / "max_energy_range_uj ").string() + " " + error);
            }
            char buffer[48];
            ssize_t read_count = read(max_fd, buffer, sizeof(buffer));
            if (read_count == -1) {
                int errsv = errno;
                const char* error = strerror(errsv);
                throw std::runtime_error(containers::string("EXTRA PROF: ERROR: ") +
                                         (dir_entry.path() / "max_energy_range_uj ").string() + " " + error);
            }
            buffer[read_count] = 0;
            close(max_fd);
            auto max_value = strtoull(buffer, nullptr, 10);
            if (max_value == 0) {
                throw std::runtime_error(containers::string("EXTRA PROF: ERROR: ") +
                                         (dir_entry.path() / "max_energy_range_uj ").string() +
                                         " could not be interpreted.");
            }

            GLOBALS.energyCounters.emplace_back(fd, max_value);
        }
    }
}

energy_uj getEnergy() {
    energy_uj energy = 0;
    char buffer[48];
    for (auto& energyCounter : GLOBALS.energyCounters) {

        ssize_t read_count = pread(energyCounter.fileDescriptor, buffer, sizeof(buffer), 0);
        if (read_count == -1) {
            int errsv = errno;
            const char* error = strerror(errsv);
            throw std::runtime_error(containers::string("EXTRA PROF: ERROR: Reading energy counter ") + error);
        }
        buffer[read_count] = 0;
        auto value = strtoull(buffer, nullptr, 10);
        if (value == 0) {
            throw std::runtime_error("EXTRA PROF: ERROR: Energy counter could not be interpreted.");
        }

        if (value < energyCounter.previousValue) {
            energyCounter.totalValue += energyCounter.maxValue - energyCounter.previousValue;
            energyCounter.previousValue = 0;
        }
        energyCounter.totalValue += value - energyCounter.previousValue;
        energyCounter.previousValue = value;
        energy += energyCounter.totalValue;
    }
    return energy;
}

void finalizeEnergy() {
    for (auto& energyCounter : GLOBALS.energyCounters) {
        close(energyCounter.fileDescriptor);
    }
    GLOBALS.energyCounters.clear();
}
#endif
} // namespace extra_prof::cpu::energy
