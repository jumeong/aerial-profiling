/*
 * profiling_stubs.cpp
 *
 * Stub definitions for symbols that are referenced by cuphy.hpp and cfo_ta_est.cu
 * (via the translation-unit include trick) but live in other cuPHY compile units
 * (nvlog.cpp, exit_handler.cpp) that we do not want to pull into this standalone build.
 *
 * These stubs satisfy the linker without pulling in the heavy nvlog dependencies.
 */

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include "exit_handler.hpp"

// -----------------------------------------------------------------------
// exit_handler stubs
// -----------------------------------------------------------------------

// Static member definition (required by exit_handler::getInstance())
exit_handler* exit_handler::instance = nullptr;

const unsigned int exit_handler::EXIT_WATCHDOG_SLEEP_SEC = 5;

// Constructor / Destructor stubs
exit_handler::exit_handler()  { exit_handler_flag = L1_RUNNING; }
exit_handler::~exit_handler() {}

// Method stubs
void exit_handler::set_exit_handler_flag(l1_state val) { exit_handler_flag = val; }

void exit_handler::test_trigger_exit(const char* file, int line, const char* info)
{
    fprintf(stderr, "[STUB] test_trigger_exit called from %s:%d : %s\n", file, line, info);
    exit(EXIT_FAILURE);
}

bool exit_handler::test_exit_in_flight()
{
    return (exit_handler_flag == L1_EXIT);
}

void exit_handler::set_exit_handler_cb(void (*exit_hdlr_cb)())
{
    exit_cb = exit_hdlr_cb;
}

uint32_t exit_handler::get_l1_state()
{
    return static_cast<uint32_t>(exit_handler_flag.load());
}

int exit_handler::start_exit_watchdog_thread(int) { return 0; }
void* exit_handler::exit_watchdog_thread_func(void*) { return nullptr; }

// Global pExitHandler reference (defined in nvlog.cpp in a full build)
exit_handler& pExitHandler = exit_handler::getInstance();
