#include "../address_mapping.h"
#include "../globals.h"

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
    uintptr_t ram;
    uintptr_t file;
};

void NameRegistry::create_address_mapping(containers::string output_dir) {
    containers::string nm_command("nm --numeric-sort --demangle ");

    // auto filename = filesystem::read_symlink("/proc/self/exe");

    std::unordered_map<containers::string, std::vector<offset_pair>> offset_mapping;

    auto main_program_handle = fopen("/proc/self/maps", "r");
    containers::string memory_sections;
    if (main_program_handle) {
        ssize_t length = 0;
        char *buffer = nullptr;
        size_t buffer_length = 0;
        offset_pair offset;
        char perm[4];
        char path[PATH_MAX];

        do {
            length = getline(&buffer, &buffer_length, main_program_handle);
            int parsed = sscanf(buffer, "%lx-%*x %4c %lx %*d:%*d %*d %[^\n]", &offset.ram, perm, &offset.file, path);

            if (parsed != 4 || path[0] != '/') {
                continue;
            }

            auto pair = offset_mapping.try_emplace(path, std::vector<offset_pair>());
            pair.first->second.push_back(offset);

            // std::cout << offset.ram << ' ' << path << '\n' << offset_ram_end << " ----\n";
        } while (length > 0);

        fclose(main_program_handle);
    }

    for (auto &[filename, offsets] : offset_mapping) {

        containers::string result_str = nm_command + filename + " 2>&1";

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
        uintptr_t adress = 0;

        // std::ofstream stream(output_dir / "symbols.txt");

        while (fgets(buffer, sizeof(buffer), fp) != nullptr) {
            int parsed = sscanf(buffer, "%lx %1c %[^\n]", &adress, &modifier, path);
            if (parsed != 3)
                continue;

            if (modifier == 't' || modifier == 'T' || modifier == 'w' || modifier == 'W') {

                auto offset_iter =
                    std::upper_bound(offsets.cbegin(), offsets.cend(), adress,
                                     [](const uintptr_t &a, const offset_pair &offset) { return a < offset.file; });
                offset_iter--;

                uintptr_t translated_adress = adress - offset_iter->file + offset_iter->ram;
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
}
}