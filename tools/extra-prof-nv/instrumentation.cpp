#include <nvToolsExt.h>
#include <stdlib.h>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <iostream>
#include <cmath>
#include <charconv>
#include <string_view>
#include <sstream>

namespace extra_prof
{

    namespace category
    {
        int RESERVED = 0;
        int SYMBOL = 1;
    }

    std::atomic<bool> initialised(false);
    std::mutex initialising;

    nvtxDomainHandle_t nvtx_domain;

#ifndef EXTRA_PROF_MAX_MAX_DEPTH
    constexpr uint32_t MAX_MAX_DEPTH = 1000;
#else
    constexpr uint32_t MAX_MAX_DEPTH = EXTRA_PROF_MAX_MAX_DEPTH;
#endif
    constexpr unsigned int MAX_ENCODED_PTR_SIZE = 11; // =1+(int)std::ceil(std::log2(std::pow(2,48)))

    thread_local unsigned int depth = 0;
    thread_local char stack[MAX_MAX_DEPTH * MAX_ENCODED_PTR_SIZE + 1] = {""};
    thread_local char *stack_ptr_stack[MAX_MAX_DEPTH + 1] = {stack};
    thread_local char *stack_ptr_last = stack + (MAX_MAX_DEPTH * MAX_ENCODED_PTR_SIZE);

    intptr_t adress_offset = INTPTR_MAX;

    uint32_t MAX_DEPTH = MAX_MAX_DEPTH;

    constexpr char UNIT_SEPERATOR = '\x1F';

    void initialize()
    {
        std::cerr << "Initialising EXTRA PROF" << std::endl;

        nvtx_domain = nvtxDomainCreateA("de.tu-darmstadt.parallel.extra_prof");
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = "extra_prof::initialize()";
        eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT32;
        eventAttrib.payload.uiValue = depth;
        nvtxDomainRangePushEx(nvtx_domain, &eventAttrib);

        const char *max_depth_str = std::getenv("EXTRA_PROF_MAX_DEPTH");
        if (max_depth_str != nullptr)
        {
            char *end;
            MAX_DEPTH = std::strtoul(max_depth_str, &end, 10);
            if (MAX_DEPTH >= MAX_MAX_DEPTH)
            {
                MAX_DEPTH = MAX_MAX_DEPTH;
            }
            std::cerr << "EXTRA PROF MAX DEPTH: " << MAX_DEPTH << std::endl;
        }

        std::string nm_command("nm --numeric-sort --demangle ");
        char filename_buffer[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", filename_buffer, PATH_MAX);
        std::string filename(filename_buffer, count);

        std::string result_str = nm_command + filename;

        const char *result = result_str.c_str();

        //printf("Command: %s", result);

        std::FILE *fp;
        /* Open the command for reading. */
        fp = popen(result, "r");
        if (fp == NULL)
        {
            std::cerr << "EXTRA PROF failed to load the symbol table" << std::endl;
            exit(1);
        }

        char buffer[2001];
        char path[2001];
        char modifier;
        unsigned long long int adress = 0;
        char base36buffer[MAX_ENCODED_PTR_SIZE + 1] = {""};

        std::stringstream stream;
        while (fgets(buffer, sizeof(buffer), fp) != NULL)
        {
            int parsed = sscanf(buffer, "%llx %1c %[^\n]", &adress, &modifier, path);
            if (parsed != 3)
                continue;

            if (modifier == 't' || modifier == 'T' || modifier == 'w' || modifier == 'W')
            {
                void *function_ptr = reinterpret_cast<void *>(adress);
                if (adress_offset == INTPTR_MAX)
                {
                    adress_offset = adress;
                    //std::cerr << "Adress offset: " << adress_offset << std::endl;
                    nvtxStringHandle_t message = nvtxDomainRegisterStringA(nvtx_domain, "EXTRA_PROF_SYMBOLS");
                    nvtxEventAttributes_t symEventAttrib = {0};
                    symEventAttrib.version = NVTX_VERSION;
                    symEventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                    symEventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
                    symEventAttrib.message.registered = message;
                    symEventAttrib.category = category::SYMBOL;
                    symEventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
                    symEventAttrib.payload.ullValue = adress_offset;
                    nvtxDomainMarkEx(nvtx_domain, &symEventAttrib);
                }

                std::to_chars_result res = std::to_chars(base36buffer, base36buffer + MAX_ENCODED_PTR_SIZE, adress - adress_offset, 36);
                if (res.ec != std::errc())
                {
                    throw res.ec;
                }
                *res.ptr = '\0';

                stream << "EP" << UNIT_SEPERATOR << base36buffer << UNIT_SEPERATOR << path;

                nvtxDomainRegisterStringA(nvtx_domain, stream.str().c_str());

                stream.str(std::string());
                stream.clear();
            }
        }

        nvtxStringHandle_t message = nvtxDomainRegisterStringA(nvtx_domain, "EXTRA_PROF_SYMBOLS_END");
        nvtxEventAttributes_t symEventAttrib = {0};
        symEventAttrib.version = NVTX_VERSION;
        symEventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        symEventAttrib.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
        symEventAttrib.message.registered = message;
        symEventAttrib.category = category::SYMBOL;
        nvtxDomainMarkEx(nvtx_domain, &symEventAttrib);

        /* close */
        pclose(fp);
        nvtxDomainRangePop(nvtx_domain);
    }
}
extern "C"
{
    void __cyg_profile_func_enter(void *this_fn,
                                  void *call_site)
    {
        if (!extra_prof::initialised.load(std::memory_order_acquire))
        {
            std::lock_guard<std::mutex> lk(extra_prof::initialising);

            if (!extra_prof::initialised.load(std::memory_order_relaxed))
            {
                extra_prof::initialize();

                extra_prof::initialised.store(true, std::memory_order_release);
            }
        }

        if (extra_prof::depth < extra_prof::MAX_DEPTH)
        {
            char *stack_ptr = extra_prof::stack_ptr_stack[extra_prof::depth];
            *stack_ptr = extra_prof::UNIT_SEPERATOR;
            std::to_chars_result res = std::to_chars(stack_ptr + 1, extra_prof::stack_ptr_last, static_cast<unsigned int>(reinterpret_cast<intptr_t>(this_fn)) - extra_prof::adress_offset, 36);
            extra_prof::stack_ptr_stack[extra_prof::depth + 1] = res.ptr;
            *res.ptr = '\0';

            //std::cerr << this_fn << ':' << extra_prof::stack << '\n';

            nvtxEventAttributes_t eventAttrib = {0};
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib.message.ascii = extra_prof::stack;

            eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT32;
            eventAttrib.payload.uiValue = extra_prof::depth;
            nvtxDomainRangePushEx(extra_prof::nvtx_domain, &eventAttrib);
        }
        extra_prof::depth++;
    }
    void __cyg_profile_func_exit(void *this_fn,
                                 void *call_site)
    {
        if (extra_prof::initialised.load(std::memory_order_relaxed))
        {
            extra_prof::depth--;
            if (extra_prof::depth < extra_prof::MAX_DEPTH)
            {
                nvtxDomainRangePop(extra_prof::nvtx_domain);
            }
        }
    }
}