#pragma once
#include "common_types.h"

#include "containers/string.h"
#include <linux/limits.h>
#include <sys/stat.h>
#include <unistd.h>
namespace extra_prof::filesystem {

inline containers::string read_symlink(const containers::string& p) {
    char buffer[PATH_MAX];
    size_t len_buffer = readlink(p.c_str(), buffer, PATH_MAX);
    if (len_buffer == -1) {
        int errsv = errno;
        const char* error = strerror(errsv);
        throw std::runtime_error(containers::string("EXTRA PROF: ERROR: Could not read symbolic link: ") + p + " " +
                                 error);
    };
    return containers::string(buffer, len_buffer);
}

inline bool is_directory(const containers::string& p) {
    struct stat path_stat;
    if (stat(p.c_str(), &path_stat) == -1) {
        int errsv = errno;
        if (errsv == ENOENT) {
            return false;
        }
        const char* error = strerror(errsv);
        throw std::runtime_error(containers::string("EXTRA PROF: ERROR: Could not check for existence of directory: ") +
                                 p + " " + error);
    };
    return S_ISDIR(path_stat.st_mode);
}

inline bool create_directory(const containers::string& p) {
    if (mkdir(p.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) != 0) {
        int errsv = errno;
        if (errsv == EEXIST) {
            return false;
        }
        const char* error = strerror(errsv);
        throw std::runtime_error(containers::string("EXTRA PROF: ERROR: Could not create folder: ") + p + " " + error);
    };
    return true;
}
} // namespace extra_prof::filesystem