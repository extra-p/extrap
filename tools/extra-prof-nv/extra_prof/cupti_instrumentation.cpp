

#include "commons.h"

#include "cupti_instrumentation.h"
#include "events.h"

#include <msgpack.hpp>

namespace extra_prof {
namespace cupti {

    void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {

        if (std::this_thread::get_id() != GLOBALS.main_thread_id) {
            if (GLOBALS.gpu.activity_thread == std::this_thread::get_id()) {
                return; // Do not show error when called from cupti activity thread
            }
            if (!GLOBALS.notMainThreadAlreadyWarned.load(std::memory_order_relaxed)) {
                std::cerr << "EXTRA PROF: WARNING: callback: Ignored additional threads.\n";
                GLOBALS.notMainThreadAlreadyWarned.store(true, std::memory_order_relaxed);
            }
            return;
        }
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

                    auto [time, call_tree_node] =
                        extra_prof::push_time(rtdata->symbolName, CallTreeNodeType::KERNEL_LAUNCH);

                    if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
                        const auto *params =
                            reinterpret_cast<const cudaLaunchKernel_v7000_params *>(rtdata->functionParams);
                        callpath_correlation.emplace(rtdata->correlationId,
                                                     CorrelationData{call_tree_node, params->func});
                    } else {
                        callpath_correlation.emplace(rtdata->correlationId, call_tree_node);
                    }
                } else {
                    auto [time, call_tree_node] = extra_prof::push_time(rtdata->functionName);
                    callpath_correlation.emplace(rtdata->correlationId, call_tree_node);
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
                        auto &ref =
                            event_stream.emplace_back(time, evt_type | EventType::START, EventStart{call_tree_node},
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
                    event_stream.emplace_back(time, evt_type | EventType::END,
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
    static const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind, CUpti_ActivityFlag flags) {
        if ((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC) {
            switch (kind) {
            case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
                return "Memcpy Async HtoD";
            case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
                return "Memcpy Async DtoH";
            case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
                return "Memcpy Async HtoA";
            case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
                return "Memcpy Async AtoH";
            case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
                return "Memcpy Async AtoA";
            case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
                return "Memcpy Async AtoD";
            case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
                return "Memcpy Async DtoA";
            case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
                return "Memcpy Async DtoD";
            case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
                return "Memcpy Async HtoH";
            case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
                return "Memcpy Async PtoP";
            default:
                break;
            }

            return "Memcpy Async Unknown";
        }
        switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "Memcpy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "Memcpy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "Memcpy HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "Memcpy AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "Memcpy AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "Memcpy AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "Memcpy DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "Memcpy DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "Memcpy HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "Memcpy PtoP";
        default:
            break;
        }

        return "Memcpy Unknown";
    }

    void process_activity_kernel(CUpti_ActivityKernel5 *record) {
#ifdef EXTRA_PROF_DEBUG
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            std::cout << "Concurrent";
        }
        std::cout << "Kernel" << record->name << " " << record->correlationId << "  from " << record->start << "  to "
                  << record->end << '\n';
#endif
        auto iterator = GLOBALS.gpu.callpath_correlation.find(record->correlationId);
        if (iterator == GLOBALS.gpu.callpath_correlation.end()) {
            std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " " << record->name << "\n";
            return;
        }
        auto &correlation_data = iterator->second;
        auto *node = correlation_data.node->findOrAddChild(record->name, CallTreeNodeType::KERNEL);
        node->setAsync(true);
        node->duration += record->end - record->start;
        node->visits++;
        int maxActiveBlocksPerMP = 0;
        GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerMP, correlation_data.function_ptr,
                                                               record->blockX * record->blockY * record->blockZ,
                                                               record->dynamicSharedMemory));

        float activeBlocks = std::min(record->gridX * record->gridY * record->gridZ,
                                      maxActiveBlocksPerMP * GLOBALS.gpu.multiProcessorCount);
        float resourceUsage = activeBlocks / (maxActiveBlocksPerMP * GLOBALS.gpu.multiProcessorCount);
        addEventPair(GLOBALS.gpu.event_stream, EventType::KERNEL, record->start, record->end, node, resourceUsage,
                     record->correlationId, record->streamId);
    }

    void process_activity_overhead(CUpti_ActivityOverhead *record) {
        const char *overheadKind;
        switch (record->overheadKind) {
        case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
            overheadKind = "OVERHEAD_DRIVER_COMPILER";
            break;

        case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
            overheadKind = "OVERHEAD_CUPTI_BUFFER_FLUSH";
            break;

        case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
            overheadKind = "OVERHEAD_CUPTI_INSTRUMENTATION";
            break;

        case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
            overheadKind = "OVERHEAD_CUPTI_RESOURCE";
            break;

        default:
            overheadKind = "OVERHEAD_UNKNOWN";
        }
#ifdef EXTRA_PROF_DEBUG
        std::cout << "OVERHEAD " << overheadKind << " from " << record->start << " to " << record->end << '\n';
#endif
        auto *node = GLOBALS.call_tree.findOrAddChild(overheadKind, CallTreeNodeType::OVERHEAD);
        node->duration += record->end - record->start;
        node->visits++;
        addEventPair(GLOBALS.gpu.event_stream, EventType::OVERHEAD, record->start, record->end, node, 0, 0, -3);
    }

    void process_activity_memcpy(CUpti_ActivityMemcpy3 *record) {
        auto *memcopy_kind = getMemcpyKindString(static_cast<CUpti_ActivityMemcpyKind>(record->copyKind),
                                                 static_cast<CUpti_ActivityFlag>(record->flags));
#ifdef EXTRA_PROF_DEBUG
        std::cout << "MEMCPY " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
        auto iterator = GLOBALS.gpu.callpath_correlation.find(record->correlationId);
        if (iterator == GLOBALS.gpu.callpath_correlation.end()) {
            std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " MEMCPY\n";
            return;
        }
        auto &correlation_data = iterator->second;
        auto *node = correlation_data.node->findOrAddChild(memcopy_kind, CallTreeNodeType::MEMCPY);
        node->duration += record->end - record->start;
        node->visits++;
        node->bytes += record->bytes;
        node->setAsync((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC);

        addEventPair(GLOBALS.gpu.event_stream, EventType::MEMCPY, record->start, record->end, node, 0,
                     record->correlationId, record->streamId);
    }

    void process_activity_memcpyp2p(CUpti_ActivityMemcpyPtoP2 *record) {
        auto *memcopy_kind = getMemcpyKindString(static_cast<CUpti_ActivityMemcpyKind>(record->copyKind),
                                                 static_cast<CUpti_ActivityFlag>(record->flags));
#ifdef EXTRA_PROF_DEBUG
        std::cout << "MEMCPY P2P " << record->correlationId << " from " << record->start << " to " << record->end
                  << '\n';
#endif
        auto iterator = GLOBALS.gpu.callpath_correlation.find(record->correlationId);
        if (iterator == GLOBALS.gpu.callpath_correlation.end()) {
            std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " MEMCPY P2P\n";
            return;
        }
        auto &correlation_data = iterator->second;
        auto *node = correlation_data.node->findOrAddChild(memcopy_kind, CallTreeNodeType::MEMCPY);
        node->duration += record->end - record->start;
        node->visits++;
        node->bytes += record->bytes;
        node->setAsync((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC);

        addEventPair(GLOBALS.gpu.event_stream, EventType::MEMCPY, record->start, record->end, node, 0,
                     record->correlationId, record->streamId);
    }

    void process_activity_memset(CUpti_ActivityMemset2 *record) {
        auto is_async = ((CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC);
        auto *memset_kind = is_async ? GLOBALS.gpu.MEMSET_ASYNC : GLOBALS.gpu.MEMSET;
#ifdef EXTRA_PROF_DEBUG
        std::cout << "MEMSET " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
        auto iterator = GLOBALS.gpu.callpath_correlation.find(record->correlationId);
        if (iterator == GLOBALS.gpu.callpath_correlation.end()) {
            std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " MEMSET\n ";
            return;
        }
        auto &correlation_data = iterator->second;
        auto *node = correlation_data.node->findOrAddChild(memset_kind, CallTreeNodeType::MEMSET);
        node->duration += record->end - record->start;
        node->visits++;
        node->bytes += record->bytes;
        node->setAsync(is_async);
        addEventPair(GLOBALS.gpu.event_stream, EventType::MEMSET, record->start, record->end, node, 0,
                     record->correlationId, record->streamId);
    }

    void CUPTIAPI on_buffer_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
        *buffer = GLOBALS.gpu.buffer_pool.get_mem();
        if (((uintptr_t)(buffer) & ((ACTIVITY_RECORD_ALIGNMENT)-1))) {
            throw std::runtime_error("EXTRA PROF: misaligned cupti buffer memory");
        }
        *size = sizeof(uint8_t) * GLOBALS.gpu.buffer_pool.size();
        *maxNumRecords = 0;
    }
    void CUPTIAPI on_buffer_complete(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size,
                                     size_t validSize) {
        GLOBALS.gpu.activity_thread = std::this_thread::get_id();

        if (GLOBALS.magic_number != 0x1A2B3C4D) {
            std::cout << "EXTRA PROF: ERROR: Global State is corrupted. \n";
            return;
        }

        CUptiResult status;
        CUpti_Activity *record = NULL;

        if (validSize > 0) {
            do {
                status = cuptiActivityGetNextRecord(buffer, validSize, &record);
                if (status == CUPTI_SUCCESS) {
                    switch (record->kind) {
                    case CUPTI_ACTIVITY_KIND_KERNEL:
                    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                        process_activity_kernel((CUpti_ActivityKernel5 *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_OVERHEAD:
                        process_activity_overhead((CUpti_ActivityOverhead *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMCPY:
                        process_activity_memcpy((CUpti_ActivityMemcpy3 *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMSET:
                        process_activity_memset((CUpti_ActivityMemset2 *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMCPY2:
                        process_activity_memcpyp2p((CUpti_ActivityMemcpyPtoP2 *)record);
                        break;
                    default:
                        std::cerr << "EXTRA PROF: WARNING: Unknown CUPTI activity " << record->kind << '\n';
                        break;
                    }
                } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                    break;
                else {
                    CUPTI_CALL(status);
                }
            } while (1);

            // report any records dropped from the queue
            size_t dropped;
            CUPTI_CALL(cuptiActivityGetNumDroppedRecords(context, streamId, &dropped));
            if (dropped != 0) {
                std::cerr << "EXTRA PROF: WARNING: Dropped " << dropped << " activity records\n";
            }
        }
        GLOBALS.gpu.buffer_pool.return_mem(buffer);
    }

    size_t event_stream_size() { return GLOBALS.gpu.event_stream.size() * (sizeof(Event) + sizeof(time_point)); }
    size_t cupti_mappings_size() {
        return GLOBALS.gpu.callpath_correlation.size() * (sizeof(uint32_t) + sizeof(CallTreeNode *));
    }

    void make_overlap(std::vector<Event *> &stack, Event *current_event, CallTreeNode *overlap, time_point duration) {
        for (auto &ke : stack) {
            if (ke == current_event) {
                continue;
            }
            if (ke->get_type() == EventType::SYNCHRONIZE || ke->get_type() == EventType::MEMMGMT) {
                continue;
            }
            auto *kernel_overlap = overlap->findOrAddChild(
                ke->start_event.node->name(), enum_cast<CallTreeNodeType>(ke->get_type()), CallTreeNodeFlags::OVERLAP);
            kernel_overlap->duration += duration;
        }
    }

    void postprocess_event_stream() {
        std::vector<Event *> sorted_stream;
        for (Event &evt : GLOBALS.gpu.event_stream) {
            sorted_stream.emplace_back(&evt);
        }
        std::stable_sort(sorted_stream.begin(), sorted_stream.end(),
                         [](Event *a, Event *b) { return (a->timestamp < b->timestamp); });

        std::vector<Event *> stack;

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

                    if (se->get_type() == EventType::KERNEL) {
                        overlap->duration += (total_MP_usage - se->resourceUsage) * duration / total_MP_usage;
                        make_overlap(stack, se, overlap, duration);
                    } else if (se->get_type() == EventType::MEMCPY) {
                        if (stack_contains_kernels > 0) {
                            overlap->duration += duration;
                            make_overlap(stack, se, overlap, duration);
                        } else if (stack_contains_memcpy > 1) {
                            overlap->duration += duration * (stack_contains_memcpy - 1) / stack_contains_memcpy;
                            make_overlap(stack, se, overlap, duration);
                        }
                    } else if (se->get_type() == EventType::MEMSET) {
                        if (stack_contains_kernels > 0 || stack_contains_memcpy > 0) {
                            overlap->duration += duration;
                            make_overlap(stack, se, overlap, duration);
                        } else if (stack_contains_memset > 1) {
                            overlap->duration += duration * (stack_contains_memset - 1) / stack_contains_memset;
                            make_overlap(stack, se, overlap, duration);
                        }
                    } else if (se->get_type() == EventType::SYNCHRONIZE || se->get_type() == EventType::MEMMGMT) {
                        if (stack_contains_kernels > 0 || stack_contains_memcpy > 0 || stack_contains_memset > 0) {
                            overlap->duration += duration;
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

};
};