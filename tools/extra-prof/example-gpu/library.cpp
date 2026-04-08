#include "library.hpp"
#include <iostream>
#include <string>

std::string values;

void dynamicHello() {
    values += "1";
    std::cout << "Hello World from dynamic library!" << values.size() << "\n";
}

void custom_exit(int argc) {
    std::cout << "Bye from dynamic library!" << values.size() << "\n";
    if (argc < 42)
        exit(0);
}