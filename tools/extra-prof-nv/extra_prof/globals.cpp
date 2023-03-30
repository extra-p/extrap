#include "globals.h"
#include <new>         // placement new
#include <type_traits> // aligned_storage

namespace extra_prof {
namespace globals_init {
    static int nifty_counter_globals; // zero initialized
    static typename std::aligned_storage<sizeof(GlobalState), alignof(GlobalState)>::type globals_buffer;
    GlobalStateInitializer::GlobalStateInitializer() {
        if (nifty_counter_globals++ == 0)
            new (&GLOBALS) GlobalState(); // placement new
    }
    GlobalStateInitializer::~GlobalStateInitializer() {
        if (--nifty_counter_globals == 0)
            (&GLOBALS)->~GlobalState();
    }
}
GlobalState &GLOBALS = reinterpret_cast<GlobalState &>(globals_init::globals_buffer);
}
