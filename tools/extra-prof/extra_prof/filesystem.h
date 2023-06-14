#pragma once
#include "common_types.h"

#include "containers/string.h"
#include <linux/limits.h>
#include <sys/stat.h>
#include <unistd.h>
namespace extra_prof::filesystem {

inline containers::string read_symlink(const containers::string &p) {
    char buffer[PATH_MAX];
    size_t len_buffer = readlink(p.c_str(), buffer, PATH_MAX);
    return containers::string(buffer, len_buffer);
}

inline bool is_directory(const containers::string &p) {
    struct stat path_stat;
    stat(p.c_str(), &path_stat);
    return S_ISDIR(path_stat.st_mode);
}

inline void create_directory(const containers::string &p) {
    if (mkdir(p.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) != 0) {
        throw containers::string("Could not create folder: ") + p;
    };
}
}