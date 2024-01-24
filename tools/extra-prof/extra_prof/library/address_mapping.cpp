#include "../address_mapping.h"
#include "../globals.h"
#include <link.h>

extra_prof::containers::string extra_prof::NameRegistry::defaultExperimentDirName() {
    time_t now = time(0);
    struct tm tstruct {};
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);

    return containers::string("extra_prof_") + buf;
}
namespace extra_prof {

struct offset_pair {
    containers::string path;
    uintptr_t offset;

    offset_pair() = default;
    offset_pair(containers::string path_, uintptr_t offset_) : path(path_), offset(offset_) {}
};

void NameRegistry::create_address_mapping(containers::string output_dir) {
#ifdef EXTRA_PROF_SCOREP_INSTRUMENTATION
    return;
#endif

    containers::string nm_command("nm --numeric-sort --demangle ");
    auto app_filename = filesystem::read_symlink("/proc/self/exe");

    std::unordered_map<containers::string, std::vector<offset_pair>> offset_mapping;
    std::vector<offset_pair> offsets;

    int result = dl_iterate_phdr(
        [](struct dl_phdr_info* info, size_t size, void* data) -> int {
            std::vector<offset_pair>* offsets_p = (std::vector<offset_pair>*)data;
            offsets_p->emplace_back(info->dlpi_name, info->dlpi_addr);
            return 0;
        },
        &offsets);

    // std::ofstream stream(output_dir + "/symbols.txt");

    for (auto&& [filepath, offset] : offsets) {
        containers::string filename = filepath;
        if (filename == "") {
            filename = app_filename;
        }

        containers::string result_str = nm_command + filename + " 2>&1";

        const char* result = result_str.c_str();

        // printf("Command: %s", result);

        FILE* fp;
        /* Open the command for reading. */
        fp = popen(result, "r");
        if (fp == nullptr) {
            throw std::runtime_error("EXTRA PROF: ERROR: Failed to load the symbol table");
        }

        char buffer[2001];
        char path[2001];
        char modifier;
        uintptr_t adress = 0;

        while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
            int parsed = sscanf(buffer, "%lx %1c %[^\n]", &adress, &modifier, path);
            if (parsed != 3)
                continue;

            if (modifier == 't' || modifier == 'T' || modifier == 'w' || modifier == 'W') {

                uintptr_t translated_adress = adress + offset;
                name_register.emplace(translated_adress, path);

                if (strcmp(path, "main") == 0) {
                    main_function_ptr = translated_adress;
                }

                // stream << translated_adress << ' ' << path << '\n';
            }
        }

        /* close */
        pclose(fp);
    }
    if (main_function_ptr == 0) {
        throw std::runtime_error("EXTRA PROF: ERROR: Failed to identify main function");
    }
}
} // namespace extra_prof
