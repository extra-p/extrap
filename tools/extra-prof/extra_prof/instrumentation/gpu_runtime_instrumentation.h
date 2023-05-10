#pragma once
#include "../globals.h"
#include "gpu_instrumentation.h"

namespace extra_prof::gpu::runtime {
void enable() {
    CUPTI_CALL(cuptiActivityRegisterCallbacks(&on_buffer_request, &on_buffer_complete));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
}

void process_activity_kernel(CUpti_ActivityKernel5 *record) {
#ifdef EXTRA_PROF_DEBUG
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        std::cout << "Concurrent";
    }
    std::cout << "Kernel" << record->name << " " << record->correlationId << "  from " << record->start << "  to "
              << record->end << '\n';
#endif
    auto correlation_data_ptr = GLOBALS.gpu.callpath_correlation.try_get(record->correlationId);
    if (correlation_data_ptr == nullptr) {
        std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " " << record->name << "\n";
        return;
    }
    auto &correlation_data = *correlation_data_ptr;
    auto *node = correlation_data.node->findOrAddChild(record->name, CallTreeNodeType::KERNEL);
    node->setAsync(true);
    auto &metrics = node->per_thread_metrics[correlation_data.thread];
    metrics.duration += record->end - record->start;
    metrics.visits++;
    int maxActiveBlocksPerMP = 0;
    GPU_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerMP, correlation_data.function_ptr,
                                                           record->blockX * record->blockY * record->blockZ,
                                                           record->dynamicSharedMemory));

    float activeBlocks =
        std::min(record->gridX * record->gridY * record->gridZ, maxActiveBlocksPerMP * GLOBALS.gpu.multiProcessorCount);
    float resourceUsage = activeBlocks / (maxActiveBlocksPerMP * GLOBALS.gpu.multiProcessorCount);
    addEventPair(GLOBALS.gpu.event_stream, EventType::KERNEL, record->start, record->end, node, correlation_data.thread,
                 resourceUsage, record->correlationId, record->streamId);
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
    auto &metrics = node->my_metrics();
    metrics.duration += record->end - record->start;
    metrics.visits++;
    addEventPair(GLOBALS.gpu.event_stream, EventType::OVERHEAD, record->start, record->end, node, pthread_self(), 0, 0,
                 -3);
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

void process_activity_memcpy(CUpti_ActivityMemcpy3 *record) {
    auto *memcopy_kind = getMemcpyKindString(static_cast<CUpti_ActivityMemcpyKind>(record->copyKind),
                                             static_cast<CUpti_ActivityFlag>(record->flags));
#ifdef EXTRA_PROF_DEBUG
    std::cout << "MEMCPY " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
    auto correlation_data_ptr = GLOBALS.gpu.callpath_correlation.try_get(record->correlationId);
    if (correlation_data_ptr == nullptr) {
        std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " MEMCOPY\n";
        return;
    }
    auto &correlation_data = *correlation_data_ptr;
    auto *node = correlation_data.node->findOrAddChild(memcopy_kind, CallTreeNodeType::MEMCPY);
    auto &metrics = node->per_thread_metrics[correlation_data.thread];
    metrics.duration += record->end - record->start;
    metrics.visits++;
    metrics.bytes += record->bytes;
    node->setAsync((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC);

    addEventPair(GLOBALS.gpu.event_stream, EventType::MEMCPY, record->start, record->end, node, correlation_data.thread,
                 0, record->correlationId, record->streamId);
}

void process_activity_memcpyp2p(CUpti_ActivityMemcpyPtoP2 *record) {
    auto *memcopy_kind = getMemcpyKindString(static_cast<CUpti_ActivityMemcpyKind>(record->copyKind),
                                             static_cast<CUpti_ActivityFlag>(record->flags));
#ifdef EXTRA_PROF_DEBUG
    std::cout << "MEMCPY P2P " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
    auto correlation_data_ptr = GLOBALS.gpu.callpath_correlation.try_get(record->correlationId);
    if (correlation_data_ptr == nullptr) {
        std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " MEMCPY\n";
        return;
    }
    auto &correlation_data = *correlation_data_ptr;
    auto *node = correlation_data.node->findOrAddChild(memcopy_kind, CallTreeNodeType::MEMCPY);
    auto &metrics = node->per_thread_metrics[correlation_data.thread];
    metrics.duration += record->end - record->start;
    metrics.visits++;
    metrics.bytes += record->bytes;
    node->setAsync((CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC);

    addEventPair(GLOBALS.gpu.event_stream, EventType::MEMCPY, record->start, record->end, node, correlation_data.thread,
                 0, record->correlationId, record->streamId);
}

void process_activity_memset(CUpti_ActivityMemset2 *record) {
    auto is_async = ((CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC & record->flags) == CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC);
    auto *memset_kind = is_async ? GLOBALS.gpu.MEMSET_ASYNC : GLOBALS.gpu.MEMSET;
#ifdef EXTRA_PROF_DEBUG
    std::cout << "MEMSET " << record->correlationId << " from " << record->start << " to " << record->end << '\n';
#endif
    auto correlation_data_ptr = GLOBALS.gpu.callpath_correlation.try_get(record->correlationId);
    if (correlation_data_ptr == nullptr) {
        std::cout << "EXTRA PROF: Unknown correlation id: " << record->correlationId << " MEMSET\n";
        return;
    }
    auto &correlation_data = *correlation_data_ptr;
    auto *node = correlation_data.node->findOrAddChild(memset_kind, CallTreeNodeType::MEMSET);
    auto &metrics = node->per_thread_metrics[correlation_data.thread];
    metrics.duration += record->end - record->start;
    metrics.visits++;
    metrics.bytes += record->bytes;
    node->setAsync(is_async);
    addEventPair(GLOBALS.gpu.event_stream, EventType::MEMSET, record->start, record->end, node, correlation_data.thread,
                 0, record->correlationId, record->streamId);
}

void CUPTIAPI on_buffer_request(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *buffer = GLOBALS.gpu.buffer_pool.get_mem();
    if (((uintptr_t)(buffer) & ((ACTIVITY_RECORD_ALIGNMENT)-1))) {
        throw std::runtime_error("EXTRA PROF: misaligned cupti buffer memory");
    }
    *size = sizeof(uint8_t) * GLOBALS.gpu.buffer_pool.size();
    *maxNumRecords = 0;
}
void CUPTIAPI on_buffer_complete(CUcontext context, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
    GLOBALS.gpu.activity_thread = pthread_self();

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
}