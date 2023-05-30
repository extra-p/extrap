#include "../globals.h"
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*P_CREATE)(pthread_t *__restrict __newthread, const pthread_attr_t *__restrict __attr,
                        void *(*__start_routine)(void *), void *__restrict __arg);

namespace extra_prof {
struct WrappedThreadArgs {
    extra_prof::ThreadState thread_state;
    void *(*start_routine)(void *);
    void *argument;
};
void *wrap_start_thread(void *wrapped_args_ptr) {

    WrappedThreadArgs *wrapped_args = reinterpret_cast<WrappedThreadArgs *>(wrapped_args_ptr);
    pthread_t new_tid = pthread_self();
    {
        extra_prof_scope sc;
        extra_prof::GLOBALS.threads.emplace(new_tid, std::move(wrapped_args->thread_state));
    }
    auto start_routine = wrapped_args->start_routine;
    auto args = wrapped_args->argument;
    delete wrapped_args;

    return start_routine(args);
}
}
#define BT_BUF_SIZE 100
extern "C" {

EXTRA_PROF_SO_EXPORT int pthread_create(pthread_t *thread, const pthread_attr_t *attr, void *(*start_routine)(void *),
                                        void *arg) {
    static void *handle = NULL;
    static P_CREATE old_create = NULL;
    if (!handle) {
        handle = dlopen("libpthread.so.0", RTLD_LAZY);
        old_create = (P_CREATE)dlsym(handle, "pthread_create");
    }
    pthread_t tid = pthread_self();

    extra_prof::WrappedThreadArgs *wrapped_arg =
        new extra_prof::WrappedThreadArgs{extra_prof::GLOBALS.my_thread_state().duplicate(), start_routine, arg};
    // int nptrs;
    // void *buffer[BT_BUF_SIZE];
    // char **strings;

    // nptrs = backtrace(buffer, BT_BUF_SIZE);
    // printf("backtrace() returned %d addresses\n", nptrs);

    // /* The call backtrace_symbols_fd(buffer, nptrs, STDOUT_FILENO)
    //    would produce similar output to the following: */

    // strings = backtrace_symbols(buffer, nptrs);
    // if (strings == NULL) {
    //     perror("backtrace_symbols");
    //     exit(-1);
    // }

    // for (int j = 0; j < nptrs; j++)
    //     printf("%s\n", strings[j]);

    // free(strings);

    // print pthread_t pid

    int result = old_create(thread, attr, &extra_prof::wrap_start_thread, wrapped_arg);

    // printf("Created new thread %lu in thread %lu \n", *thread, tid);
    //  print thread pid

    return result;
}
}