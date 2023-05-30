#pragma once
#include "common_types.h"

#include "containers/string.h"
#include "filesystem.h"
#include "globals.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <msgpack.hpp>
#include <mutex>
#include <vector>
namespace extra_prof {

// TODO write name registry class

inline containers::string currentDateTime() {
    time_t now = time(0);
    struct tm tstruct {};
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);

    return buf;
}

inline void create_address_mapping(containers::string output_dir) {
    auto &name_register = GLOBALS.name_register;
    auto &main_function_ptr = GLOBALS.main_function_ptr;
    containers::string nm_command("nm --numeric-sort --demangle ");

    auto filename = filesystem::read_symlink("/proc/self/exe");

    containers::string result_str = nm_command + filename;

    const char *result = result_str.c_str();

    // printf("Command: %s", result);

    FILE *fp;
    /* Open the command for reading. */
    fp = popen(result, "r");
    if (fp == nullptr) {
        std::cerr << "EXTRA PROF: ERROR: Failed to load the symbol table" << std::endl;
        exit(1);
    }

    char buffer[2001];
    char path[2001];
    char modifier;
    unsigned long long adress = 0;

    // std::ofstream stream(output_dir / "symbols.txt");

    while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
        int parsed = sscanf(buffer, "%llx %1c %[^\n]", &adress, &modifier, path);
        if (parsed != 3)
            continue;

        if (modifier == 't' || modifier == 'T' || modifier == 'w' || modifier == 'W') {

            name_register.emplace(adress, path);

            if (strcmp(path, "main") == 0) {
                main_function_ptr = adress;
            }

            // stream << adress << ' ' << path << '\n';
        }
    }

    /* close */
    pclose(fp);
}
}