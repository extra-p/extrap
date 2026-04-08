#include "../globals.h"
#include <dlfcn.h>
#include <execinfo.h>

namespace extra_prof {
void finalize_on_exit();
}

typedef __attribute__((noreturn)) void (*exit_function_type)(int exit_code);

extern "C" {

EXTRA_PROF_SO_EXPORT __attribute__((noreturn)) void exit(int exit_code) {
    static void* handle = NULL;
    static exit_function_type old_exit = NULL;
    if (!handle) {
        handle = dlopen("libc.so.6", RTLD_LAZY);
        old_exit = (exit_function_type)dlsym(handle, "exit");
    }
    if (extra_prof::extra_prof_scope_counter == 0) {
        extra_prof::finalize_on_exit();
    }
    old_exit(exit_code);
}
}