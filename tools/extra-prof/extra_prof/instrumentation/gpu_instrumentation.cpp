

#include "gpu_instrumentation.h"
#include "commons.h"

#include "gpu_hwc_instrumentation.h"
#include "gpu_runtime_instrumentation.h"

#include "../events.h"
#include <msgpack.hpp>

namespace extra_prof {
namespace cupti {

    void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
        pthread_t this_thread = pthread_self();
        if (GLOBALS.gpu.activity_thread == this_thread) {
            return; // Do not show error when called from cupti activity thread
        }
        // if (std::this_thread::get_id() != GLOBALS.main_thread_id) {

        //     if (!GLOBALS.notMainThreadAlreadyWarned.load(std::memory_order_relaxed)) {
        //         std::cerr << "EXTRA PROF: WARNING: callback: Ignored additional threads.\n";
        //         GLOBALS.notMainThreadAlreadyWarned.store(true, std::memory_order_relaxed);
        //     }
        //     return;
        // }
        auto &callpath_correlation = GLOBALS.gpu.callpath_correlation;
        auto &event_stream = GLOBALS.gpu.event_stream;
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            const CUpti_CallbackData *rtdata = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
            auto cbid_is_synchronization = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020 ||
                                           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaThreadSynchronize_v3020 ||
                                           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaEventSynchronize_v3020 ||
                                           cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020;
            auto cbid_is_mem_mgmt = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaFreeArray_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaFreeHost_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3D_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc3DArray_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocMipmappedArray_v5000 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocManaged_v6000 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocPitch_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocArray_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMallocHost_v3020 ||
                                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaHostAlloc_v3020;
            if (rtdata->callbackSite == CUPTI_API_ENTER) {
                if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
                    if (GLOBALS.gpu.onKernelLaunch != nullptr) {
                        (*GLOBALS.gpu.onKernelLaunch)(rtdata);
                    }
                    auto [time, call_tree_node] =
                        extra_prof::push_time(rtdata->symbolName, CallTreeNodeType::KERNEL_LAUNCH);
                    std::cout << "Kernel " << rtdata->symbolName << " Id: " << rtdata->correlationId << '\n';
                    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                        const auto *params =
                            reinterpret_cast<const cudaLaunchKernel_v7000_params *>(rtdata->functionParams);
                        callpath_correlation.emplace(rtdata->correlationId,
                                                     CorrelationData{call_tree_node, this_thread, params->func});
                    } else {
                        callpath_correlation.emplace(rtdata->correlationId,
                                                     CorrelationData{call_tree_node, this_thread});
                    }
                } else {
                    auto [time, call_tree_node] = extra_prof::push_time(rtdata->functionName);
                    callpath_correlation.emplace(rtdata->correlationId, CorrelationData{call_tree_node, this_thread});
                    auto evt_type = EventType::NONE;
                    int stream_id = 0;
                    if (cbid_is_synchronization) {
                        evt_type = EventType::SYNCHRONIZE;
                        stream_id = -1;
                    } else if (cbid_is_mem_mgmt) {
                        evt_type = EventType::MEMMGMT;
                        stream_id = -2;
                    }
                    if (evt_type != EventType::NONE) {
                        auto &ref = event_stream.emplace(time, evt_type | EventType::START,
                                                         EventStart{call_tree_node, this_thread},
                                                         static_cast<uint16_t>(stream_id));
                        *rtdata->correlationData = reinterpret_cast<uint64_t>(&ref);
                    }
                }

#ifdef EXTRA_PROF_DEBUG
                std::cout << "RT API START: " << rtdata->functionName << " " << rtdata->correlationId << '\n';
#endif
            } else if (rtdata->callbackSite == CUPTI_API_EXIT) {
                time_point time = 0;
                if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
                    time = extra_prof::pop_time(rtdata->symbolName);
                } else {
                    time = extra_prof::pop_time(rtdata->functionName);
                }
                auto evt_type = EventType::NONE;
                int stream_id = 0;
                if (cbid_is_synchronization) {
                    evt_type = EventType::SYNCHRONIZE;
                    stream_id = -1;
                } else if (cbid_is_mem_mgmt) {
                    evt_type = EventType::MEMMGMT;
                    stream_id = -2;
                }
                if (evt_type != EventType::NONE) {
                    event_stream.emplace(time, evt_type | EventType::END,
                                         EventEnd{reinterpret_cast<Event *>(*rtdata->correlationData)},
                                         static_cast<uint16_t>(stream_id));
                }

#ifdef EXTRA_PROF_DEBUG
                std::cout << "RT API START: " << rtdata->functionName << " " << rtdata->correlationId << '\n';
#endif
            } else {
                throw std::runtime_error("EXTRA PROF: Unknown rtdata->callbackSite value");
            }
        }
    }

    size_t event_stream_size() {
        return GLOBALS.gpu.event_stream.estimate_size() * (sizeof(Event) + sizeof(time_point));
    }

    size_t cupti_mappings_size() {
        return GLOBALS.gpu.callpath_correlation.size() * (sizeof(uint32_t) + sizeof(CallTreeNode *));
    }

    void make_overlap(std::vector<const Event *> &stack, const Event *current_event, CallTreeNode *overlap,
                      time_point duration) {
        for (auto &ke : stack) {
            if (ke == current_event) {
                continue;
            }
            if (ke->get_type() == EventType::SYNCHRONIZE || ke->get_type() == EventType::MEMMGMT) {
                continue;
            }
            auto *kernel_overlap = overlap->findOrAddChild(
                ke->start_event.node->name(), enum_cast<CallTreeNodeType>(ke->get_type()), CallTreeNodeFlags::OVERLAP);
            kernel_overlap->per_thread_metrics[ke->start_event.thread].duration += duration;
        }
    }

    void postprocess_event_stream() {
        std::vector<const Event *> sorted_stream;
        sorted_stream.reserve(GLOBALS.gpu.event_stream.estimate_size());
        auto end = GLOBALS.gpu.event_stream.cend();
        for (auto iter = GLOBALS.gpu.event_stream.cbegin(); iter != end; ++iter) {
            sorted_stream.emplace_back(&(*iter));
        }

        // for (auto event : sorted_stream) {
        //     std::cout << event->start_event.node->name() << '\n';
        // }
        std::stable_sort(sorted_stream.begin(), sorted_stream.end(),
                         [](const Event *a, const Event *b) { return (a->timestamp < b->timestamp); });

        std::vector<const Event *> stack;

        // TODO maybe add malloc/free

        time_point previous_timestamp = 0;

        for (auto &event_ptr : sorted_stream) {
            auto &event = *event_ptr;
            if (!stack.empty()) {
                float total_MP_usage = 0;
                uint32_t stack_contains_kernels = 0;
                uint32_t stack_contains_memcpy = 0;
                uint32_t stack_contains_memset = 0;
                for (auto &se : stack) {
                    if (se->get_type() == EventType::KERNEL) {
                        stack_contains_kernels++;
                        total_MP_usage += se->resourceUsage;
                    } else if (se->get_type() == EventType::MEMCPY) {
                        stack_contains_memcpy++;
                    } else if (se->get_type() == EventType::MEMSET) {
                        stack_contains_memset++;
                    }
                }
                for (auto &se : stack) {
                    time_point duration = event.timestamp - previous_timestamp;
                    CallTreeNode *overlap = se->start_event.node->findOrAddChild(
                        GLOBALS.gpu.OVERLAP, CallTreeNodeType::OVERLAP, CallTreeNodeFlags::OVERLAP);
                    auto &overlap_metrics = overlap->per_thread_metrics[se->start_event.thread];
                    if (se->get_type() == EventType::KERNEL) {
                        overlap_metrics.duration += (total_MP_usage - se->resourceUsage) * duration / total_MP_usage;
                        make_overlap(stack, se, overlap, duration);
                    } else if (se->get_type() == EventType::MEMCPY) {
                        if (stack_contains_kernels > 0) {
                            overlap_metrics.duration += duration;
                            make_overlap(stack, se, overlap, duration);
                        } else if (stack_contains_memcpy > 1) {
                            overlap_metrics.duration += duration * (stack_contains_memcpy - 1) / stack_contains_memcpy;
                            make_overlap(stack, se, overlap, duration);
                        }
                    } else if (se->get_type() == EventType::MEMSET) {
                        if (stack_contains_kernels > 0 || stack_contains_memcpy > 0) {
                            overlap_metrics.duration += duration;
                            make_overlap(stack, se, overlap, duration);
                        } else if (stack_contains_memset > 1) {
                            overlap_metrics.duration += duration * (stack_contains_memset - 1) / stack_contains_memset;
                            make_overlap(stack, se, overlap, duration);
                        }
                    } else if (se->get_type() == EventType::SYNCHRONIZE || se->get_type() == EventType::MEMMGMT) {
                        if (stack_contains_kernels > 0 || stack_contains_memcpy > 0 || stack_contains_memset > 0) {
                            overlap_metrics.duration += duration;
                            make_overlap(stack, se, overlap, duration);
                        }
                    }
                }
            }

            if (event.is_start()) {
                stack.push_back(&event);
            } else {
                stack.erase(std::remove(stack.begin(), stack.end(), event.end_event.start), stack.end());
            }
            previous_timestamp = event.timestamp;
        }
    }

    inline void init() {

        const char *EXTRA_PROF_CUPTI_BUFFER_SIZE = getenv("EXTRA_PROF_CUPTI_BUFFER_SIZE");
        if (EXTRA_PROF_CUPTI_BUFFER_SIZE != nullptr) {
            size_t number = std::stoul(EXTRA_PROF_CUPTI_BUFFER_SIZE);
            if (number < 1024) {

                number = 1024;
            }
            std::cerr << "EXTRA PROF: CUPTI BUFFER SIZE: " << number << '\n';
            GLOBALS.gpu.buffer_pool.initial_buffer_resize(number);
        }

        CUPTI_CALL(cuptiSubscribe(&GLOBALS.gpu.subscriber, (CUpti_CallbackFunc)&on_callback, nullptr));

        cudaDeviceProp prop;
        GPU_CALL(cudaGetDeviceProperties(&prop, 0));
        GLOBALS.gpu.multiProcessorCount = prop.multiProcessorCount;

        const char *EXTRA_PROF_GPU_METRICS = getenv("EXTRA_PROF_GPU_METRICS");
        if (EXTRA_PROF_GPU_METRICS != nullptr) {
            extra_prof::gpu::hwc::init();
        } else {
            extra_prof::gpu::runtime::enable();
        }

        CUPTI_CALL(cuptiEnableDomain(1, GLOBALS.gpu.subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
    }
    void finalize() {
        cudaDeviceSynchronize();
        if (!GLOBALS.gpu.metricNames.empty()) {
            extra_prof::gpu::hwc::finalize();
        } else {
            cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        }
        cuptiFinalize();

        extra_prof::gpu::hwc::postprocess_counter_data();
        postprocess_event_stream();
    }
};
};