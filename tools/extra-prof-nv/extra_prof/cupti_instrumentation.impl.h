

#include "commons.h"

#include "cupti_instrumentation.h"
#include "events.h"

#include <msgpack.hpp>

namespace extra_prof {
namespace cupti {

    const char *MEMSET = "Memset";
    const char *MEMSET_ASYNC = "Memset Async";

    const char *OVERLAP = "OVERLAP";

    std::map<time_point, Event> event_stream;

    std::unordered_map<uint32_t, CallTreeNode *> callpath_correlation;

    void CUPTIAPI on_callback(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
        if (std::this_thread::get_id() != extra_prof::main_thread_id) {
            std::cerr << "EXTRA PROF: WARNING: callback: Ignored additional threads.\n";
            return;
        }
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
                    callpath_correlation[rtdata->correlationId] = call_tree_node;

                } else {
                    auto [time, call_tree_node] = extra_prof::push_time(rtdata->functionName);
                    callpath_correlation[rtdata->correlationId] = call_tree_node;
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
                        auto [map_iter, created] =
                            event_stream.try_emplace(time, evt_type | EventType::START, EventStart{call_tree_node},
                                                     static_cast<uint16_t>(stream_id));
                        *rtdata->correlationData = reinterpret_cast<uint64_t>(&map_iter->second);
                    }
                }

#ifdef EXTRA_PROF_DEBUG
                std::cout << "RT API START: " << rtdata->functionName << " " << rtdata->correlationId << '\n';
#endif
            } else if (rtdata->callbackSite == CUPTI_API_EXIT) {
                auto time = extra_prof::pop_time();
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
                    event_stream.try_emplace(time, evt_type | EventType::END,
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

    void process_activity_kernel(CUpti_ActivityKernel7 *record) {
#ifdef EXTRA_PROF_DEBUG
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            std::cout << "Concurrent";
        }
        std::cout << "Kernel" << record->name << " " << record->correlationId << "  from " << record->start << "  to "
                  << record->end << '\n';
#endif
        auto *node =
            callpath_correlation[record->correlationId]->findOrAddChild(record->name, CallTreeNodeType::KERNEL);
        node->setAsync(true);
        node->duration += record->end - record->start;
        node->visits++;
        uint32_t threads = record->blockX * record->blockY * record->blockZ;
        threads *= record->gridX * record->gridY * record->gridZ;
        threads = std::min(threads, cupti::maxThreadsPerMultiProcessor * cupti::multiProcessorCount);
        addEventPair(event_stream, EventType::KERNEL, record->start, record->end, node, threads, record->correlationId,
                     record->streamId);
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
        auto *node = call_tree.findOrAddChild(overheadKind, CallTreeNodeType::OVERHEAD);
        node->duration += record->end - record->start;
        node->visits++;
        addEventPair(event_stream, EventType::OVERHEAD, record->start, record->end, node, 0, 0, -3);
    }

    void process_activity_memcpy(CUpti_ActivityMemcpy5 *record) {
        auto *memcopy_kind = getMemcpyKindString(static_cast<CUpti_ActivityMemcpyKind>(record->copyKind),
                                                 static_cast<CUpti_ActivityFlag>(record->flags));
#ifdef EXTRA_PROF_DEBUG
        std::cout << "MEMCPY " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
        auto *node =
            callpath_correlation[record->correlationId]->findOrAddChild(memcopy_kind, CallTreeNodeType::MEMCPY);
        node->duration += record->end - record->start;
        node->visits++;
        node->bytes += record->bytes;
        node->setAsync((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC);

        addEventPair(event_stream, EventType::MEMCPY, record->start, record->end, node, 0, record->correlationId,
                     record->streamId);
    }

    void process_activity_memcpyp2p(CUpti_ActivityMemcpyPtoP4 *record) {
        auto *memcopy_kind = getMemcpyKindString(static_cast<CUpti_ActivityMemcpyKind>(record->copyKind),
                                                 static_cast<CUpti_ActivityFlag>(record->flags));
#ifdef EXTRA_PROF_DEBUG
        std::cout << "MEMCPY P2P " << record->correlationId << " from " << record->start << " to " << record->end
                  << '\n';
#endif
        auto *node =
            callpath_correlation[record->correlationId]->findOrAddChild(memcopy_kind, CallTreeNodeType::MEMCPY);
        node->duration += record->end - record->start;
        node->visits++;
        node->bytes += record->bytes;
        node->setAsync((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC);

        addEventPair(event_stream, EventType::MEMCPY, record->start, record->end, node, 0, record->correlationId,
                     record->streamId);
    }

    void process_activity_memset(CUpti_ActivityMemset4 *record) {
        auto is_async = ((CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC);
        auto *memset_kind = is_async ? MEMSET_ASYNC : MEMSET;
#ifdef EXTRA_PROF_DEBUG
        std::cout << "MEMSET " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
        auto *node = callpath_correlation[record->correlationId]->findOrAddChild(memset_kind, CallTreeNodeType::MEMSET);
        node->duration += record->end - record->start;
        node->visits++;
        node->bytes += record->bytes;
        node->setAsync(is_async);
        addEventPair(event_stream, EventType::MEMSET, record->start, record->end, node, 0, record->correlationId,
                     record->streamId);
    }

    void CUPTIAPI on_buffer_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
        *buffer = buffer_pool.get_mem();
        if (((uintptr_t)(buffer) & ((ACTIVITY_RECORD_ALIGNMENT)-1))) {
            throw std::runtime_error("EXTRA PROF: misaligned cupti buffer memory");
        }
        *size = sizeof(uint8_t) * buffer_pool.size();
        *maxNumRecords = 0;
    }
    void CUPTIAPI on_buffer_complete(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size,
                                     size_t validSize) {
        CUptiResult status;
        CUpti_Activity *record = NULL;

        if (validSize > 0) {
            do {
                status = cuptiActivityGetNextRecord(buffer, validSize, &record);
                if (status == CUPTI_SUCCESS) {
                    switch (record->kind) {
                    case CUPTI_ACTIVITY_KIND_KERNEL:
                    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
                        process_activity_kernel((CUpti_ActivityKernel7 *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_OVERHEAD:
                        process_activity_overhead((CUpti_ActivityOverhead *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMCPY:
                        process_activity_memcpy((CUpti_ActivityMemcpy5 *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMSET:
                        process_activity_memset((CUpti_ActivityMemset4 *)record);
                        break;
                    case CUPTI_ACTIVITY_KIND_MEMCPY2:
                        process_activity_memcpyp2p((CUpti_ActivityMemcpyPtoP4 *)record);
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
        buffer_pool.return_mem(buffer);
    }

    size_t event_stream_size() { return cupti::event_stream.size() * (sizeof(Event) + sizeof(time_point)); }
    size_t cupti_mappings_size() { return callpath_correlation.size() * (sizeof(uint32_t) + sizeof(CallTreeNode *)); }

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
        std::vector<Event *> stack;

        // TODO maybe add malloc/free

        time_point previous_timestamp = 0;

        for (auto &[timestamp, event] : event_stream) {
            // auto &event = i.second;
            if (!stack.empty()) {
                uint32_t total_threads = 0;
                uint32_t stack_contains_kernels = 0;
                uint32_t stack_contains_memcpy = 0;
                uint32_t stack_contains_memset = 0;
                for (auto &se : stack) {
                    if (se->get_type() == EventType::KERNEL) {
                        stack_contains_kernels++;
                        total_threads += se->threads;
                    } else if (se->get_type() == EventType::MEMCPY) {
                        stack_contains_memcpy++;
                    } else if (se->get_type() == EventType::MEMSET) {
                        stack_contains_memset++;
                    }
                }
                for (auto &se : stack) {
                    time_point duration = timestamp - previous_timestamp;
                    CallTreeNode *overlap = se->start_event.node->findOrAddChild(OVERLAP, CallTreeNodeType::OVERLAP,
                                                                                 CallTreeNodeFlags::OVERLAP);

                    if (se->get_type() == EventType::KERNEL) {
                        overlap->duration += (total_threads - se->threads) * duration / total_threads;
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
            previous_timestamp = timestamp;
        }
    }

};
};