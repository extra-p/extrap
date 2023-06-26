#include "../globals.h"
#include <dlfcn.h>
#include <execinfo.h>

namespace extra_prof {
void finalize_on_exit();
}
extern "C" {

EXTRA_PROF_SO_EXPORT __attribute__((noreturn)) void exit(int exit_code) {
    static void *handle = NULL;
    static __attribute__((noreturn)) void (*old_exit)(int) = NULL;
    if (!handle) {
        handle = dlopen("libc.so.6", RTLD_LAZY);
        old_exit = (void (*)(int))dlsym(handle, "exit");
    }
    if (extra_prof::extra_prof_scope_counter == 0) {
        extra_prof::finalize_on_exit();
    }
    old_exit(exit_code);
}
}