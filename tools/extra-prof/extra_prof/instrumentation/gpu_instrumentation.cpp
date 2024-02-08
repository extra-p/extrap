

#include "gpu_instrumentation.h"
#include "commons.h"
#include "start_end.h"

#include "gpu_energy.h"
#include "gpu_hwc_instrumentation.h"
#include "gpu_runtime_instrumentation.h"

#include "../events.h"
#include <msgpack.hpp>

namespace extra_prof {
namespace cupti {

    void CUPTIAPI on_callback(void* userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
        if (extra_prof_scope_counter > 0) {
            return; // Do not register activity if called from extra prof scope
        }
        extra_prof_scope sc;

        pthread_t this_thread = pthread_self();

        auto& callpath_correlation = GLOBALS.gpu.callpath_correlation;
        auto& event_stream = GLOBALS.gpu.event_stream;
        if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
            const CUpti_CallbackData* rtdata = reinterpret_cast<const CUpti_CallbackData*>(cbdata);
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
            auto cbid_is_memset = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset_v3020 ||
                                  cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020 ||
                                  cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset2D_v3020 ||
                                  cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset2DAsync_v3020 ||
                                  cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset3D_v3020 ||
                                  cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemset3DAsync_v3020;

            if (rtdata->callbackSite == CUPTI_API_ENTER) {
                if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000 ||
                    cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) {
                    auto onKernelLaunch = GLOBALS.gpu.onKernelLaunch.load(std::memory_order_acquire);
                    if (onKernelLaunch != nullptr) {
                        (*onKernelLaunch)(rtdata);
                    }
                    auto [time, call_tree_node] =
                        extra_prof::push_time(rtdata->symbolName, CallTreeNodeType::KERNEL_LAUNCH);

                    // std::cout << "Kernel " << rtdata->symbolName << " Id: " << rtdata->correlationId << '\n';
                    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                        const auto* params =
                            reinterpret_cast<const cudaLaunchKernel_v7000_params*>(rtdata->functionParams);
                        callpath_correlation.emplace(rtdata->correlationId,
                                                     CorrelationData{call_tree_node, this_thread, params->func});
                    } else {
                        callpath_correlation.emplace(rtdata->correlationId,
                                                     CorrelationData{call_tree_node, this_thread});
                    }
                } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020) {
                    extra_prof::finalize_on_exit();
                    throw std::runtime_error(
                        "EXTRA PROF: Error: Device reset is not supported, because it destroys the profiling "
                        "environment. Your measurements up to this point are saved in the profile.");
                } else {
                    // if (cbid_is_memset) {
                    //  Cuda memset also launches a kernel, which needs to be registered for correctly assigning GPU
                    //  HWC values.
                    //  IMPORTANT: Apparently this is not true and created some issues with the correct mapping
                    //  between range and correlationId if (GLOBALS.gpu.onKernelLaunch != nullptr) {
                    //      (*GLOBALS.gpu.onKernelLaunch)(rtdata);
                    //  }
                    //}
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
                        auto& ref = event_stream.emplace(time, evt_type | EventType::START,
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
                                         EventEnd{reinterpret_cast<Event*>(*rtdata->correlationData)},
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
        return GLOBALS.gpu.callpath_correlation.size() * (sizeof(uint32_t) + sizeof(CallTreeNode*));
    }

    void make_overlap(std::vector<const Event*>& stack, const Event* current_event, CallTreeNode* overlap,
                      time_point duration) {
        for (auto& ke : stack) {
            if (ke == current_event) {
                continue;
            }
            if (ke->get_type() == EventType::SYNCHRONIZE || ke->get_type() == EventType::MEMMGMT) {
                continue;
            }
            auto* kernel_overlap =
                overlap->findOrAddChild(ke->start_event.node->region_type, ke->start_event.node->region,
                                        enum_cast<CallTreeNodeType>(ke->get_type()), CallTreeNodeFlags::OVERLAP);
            kernel_overlap->update_metrics(ke->start_event.thread,
                                           [&](auto& metrics) { metrics.duration += duration; });
        }
    }

    void postprocess_event_stream() {
        std::vector<const Event*> sorted_stream;
        sorted_stream.reserve(GLOBALS.gpu.event_stream.estimate_size());
        auto end = GLOBALS.gpu.event_stream.cend();
        for (auto iter = GLOBALS.gpu.event_stream.cbegin(); iter != end; ++iter) {
            auto& event = *iter;
            sorted_stream.emplace_back(&event);
        }

        std::stable_sort(sorted_stream.begin(), sorted_stream.end(),
                         [](const Event* a, const Event* b) { return (a->timestamp < b->timestamp); });

        std::vector<const Event*> stack;

        time_point previous_timestamp = 0;
        pthread_t my_thread = pthread_self();

        for (auto& event_ptr : sorted_stream) {
            auto& event = *event_ptr;
            if (!stack.empty()) {
                float total_MP_usage = 0;
                uint32_t stack_contains_kernels = 0;
                uint32_t stack_contains_memcpy = 0;
                uint32_t stack_contains_memset = 0;
                for (auto& se : stack) {
                    if (se->get_type() == EventType::KERNEL) {
                        stack_contains_kernels++;
                        total_MP_usage += se->resourceUsage;
                    } else if (se->get_type() == EventType::MEMCPY) {
                        stack_contains_memcpy++;
                    } else if (se->get_type() == EventType::MEMSET) {
                        stack_contains_memset++;
                    }
                }
                for (auto& se : stack) {
                    time_point duration = event.timestamp - previous_timestamp;
                    CallTreeNode* event_node = se->start_event.node;
                    if (event_node->owner_thread != my_thread) {
                        event_node = event_node->findOrAddPeer(false);
                    }
                    CallTreeNode* overlap =
                        event_node->findOrAddChild(RegionType::NAMED_REGION, toRegionID(GLOBALS.gpu.OVERLAP),
                                                   CallTreeNodeType::OVERLAP, CallTreeNodeFlags::OVERLAP);
                    overlap->update_metrics(se->start_event.thread, [&](auto& overlap_metrics) {
                        if (se->get_type() == EventType::KERNEL) {
                            overlap_metrics.duration +=
                                (total_MP_usage - se->resourceUsage) * duration / total_MP_usage;
                            make_overlap(stack, se, overlap, duration);
                        } else if (se->get_type() == EventType::MEMCPY) {
                            if (stack_contains_kernels > 0) {
                                overlap_metrics.duration += duration;
                                make_overlap(stack, se, overlap, duration);
                            } else if (stack_contains_memcpy > 1) {
                                overlap_metrics.duration +=
                                    duration * (stack_contains_memcpy - 1) / stack_contains_memcpy;
                                make_overlap(stack, se, overlap, duration);
                            }
                        } else if (se->get_type() == EventType::MEMSET) {
                            if (stack_contains_kernels > 0 || stack_contains_memcpy > 0) {
                                overlap_metrics.duration += duration;
                                make_overlap(stack, se, overlap, duration);
                            } else if (stack_contains_memset > 1) {
                                overlap_metrics.duration +=
                                    duration * (stack_contains_memset - 1) / stack_contains_memset;
                                make_overlap(stack, se, overlap, duration);
                            }
                        } else if (se->get_type() == EventType::SYNCHRONIZE || se->get_type() == EventType::MEMMGMT) {
                            if (stack_contains_kernels > 0 || stack_contains_memcpy > 0 || stack_contains_memset > 0) {
                                overlap_metrics.duration += duration;
                                make_overlap(stack, se, overlap, duration);
                            }
                        }
                    });
                }
            }

            if (event.is_start()) {
                stack.push_back(&event);
            } else {
                stack.erase(std::remove(stack.begin(), stack.end(), event.end_event.start), stack.end());
                auto& start_event = event.end_event.start;
#ifdef EXTRA_PROF_ENERGY
                if (start_event->start_event.thread == GLOBALS.main_thread) {
                    start_event->start_event.node->energy_gpu +=
                        GLOBALS.gpu.energySampler.getEnergy(start_event->timestamp, event.timestamp);
                }
#endif
            }
            previous_timestamp = event.timestamp;
        }
    }

    inline void init() {

        const char* EXTRA_PROF_CUPTI_BUFFER_SIZE = getenv("EXTRA_PROF_CUPTI_BUFFER_SIZE");
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

        const char* EXTRA_PROF_GPU_METRICS = getenv("EXTRA_PROF_GPU_METRICS");
        if (EXTRA_PROF_GPU_METRICS != nullptr) {
            extra_prof::gpu::hwc::init();
        } else {
#ifdef EXTRA_PROF_ENERGY
            GLOBALS.gpu.energySampler.start();
#endif
            extra_prof::gpu::runtime::init();
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
#ifdef EXTRA_PROF_ENERGY
        GLOBALS.gpu.energySampler.stop();
#endif

        if (!GLOBALS.gpu.metricNames.empty()) {
            extra_prof::gpu::hwc::postprocess_counter_data();
        }
        postprocess_event_stream();
        cudaDeviceReset();
    }
}; // namespace cupti
}; // namespace extra_prof