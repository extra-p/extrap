#include "library.hpp"
#include <iostream>

std::string values;

void dynamicHello() {
    values += "1";
    std::cout << "Hello World from dynamic library!" << values.size() << "\n";
}
