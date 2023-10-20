#include "../common_types.h"
#include <fstream>
#include <iostream>
namespace extra_prof::cpu::energy {

energy_uj getEnergy() {
    energy_uj energy = 0;
    energy_uj energy_temp;
    std::ifstream myfile;
    myfile.rdbuf()->pubsetbuf(0, 0);
    myfile.open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    myfile >> energy_temp;
    energy += energy_temp;
    myfile.close();
    myfile.open("/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj");
    myfile >> energy_temp;
    energy += energy_temp;
    myfile.close();
    myfile.open("/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj");
    myfile >> energy_temp;
    energy += energy_temp;
    myfile.close();
    myfile.open("/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj");
    myfile >> energy_temp;
    energy += energy_temp;
    myfile.close();
    return energy;
}

} // namespace extra_prof::cpu::energy
