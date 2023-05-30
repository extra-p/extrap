#pragma once
#include "../common_types.h"

#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <nvperf_cuda_host.h>
#include <nvperf_host.h>
#include <nvperf_target.h>

#include <string>
#include <vector>

#define NVPW_CALL(apiFuncCall)                                                                                         \
    do {                                                                                                               \
        NVPA_Status _status = apiFuncCall;                                                                             \
        if (_status != NVPA_STATUS_SUCCESS) {                                                                          \
            fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n", __FILE__, __LINE__, #apiFuncCall,     \
                    _status);                                                                                          \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

namespace extra_prof::gpu::hwc {

bool GetRawMetricRequests(NVPA_MetricsContext *pMetricsContext, std::vector<containers::string> metricNames,
                          std::vector<NVPA_RawMetricRequest> &rawMetricRequests,
                          std::vector<containers::string> &temp) {
    containers::string reqName;
    bool isolated = true;
    bool keepInstances = true;

    for (auto &metricName : metricNames) {
        /* Bug in collection with collection of metrics without instances, keep it to true*/
        keepInstances = true;
        NVPW_MetricsContext_GetMetricProperties_Begin_Params getMetricPropertiesBeginParams = {
            NVPW_MetricsContext_GetMetricProperties_Begin_Params_STRUCT_SIZE,
            .pMetricsContext = pMetricsContext,
            .pMetricName = metricName.c_str(),
        };

        NVPW_CALL(NVPW_MetricsContext_GetMetricProperties_Begin(&getMetricPropertiesBeginParams));

        for (const char **ppMetricDependencies = getMetricPropertiesBeginParams.ppRawMetricDependencies;
             *ppMetricDependencies; ++ppMetricDependencies) {
            temp.push_back(*ppMetricDependencies);
        }

        NVPW_MetricsContext_GetMetricProperties_End_Params getMetricPropertiesEndParams = {
            NVPW_MetricsContext_GetMetricProperties_End_Params_STRUCT_SIZE,
            .pMetricsContext = pMetricsContext,
        };
        NVPW_CALL(NVPW_MetricsContext_GetMetricProperties_End(&getMetricPropertiesEndParams));
    }

    for (auto &rawMetricName : temp) {
        NVPA_RawMetricRequest metricRequest = {
            NVPA_RAW_METRIC_REQUEST_STRUCT_SIZE,
            .pMetricName = rawMetricName.c_str(),
            .isolated = isolated,
            .keepInstances = keepInstances,
        };
        rawMetricRequests.push_back(metricRequest);
    }

    return true;
}

void GetConfigImage(const char *chipName, std::vector<NVPA_RawMetricRequest> &rawMetricRequests,
                    std::vector<uint8_t> &configImage) {

#if CUDART_VERSION < 11040
    NVPA_RawMetricsConfigOptions metricsConfigOptions = {NVPA_RAW_METRICS_CONFIG_OPTIONS_STRUCT_SIZE};
#else
    NVPW_CUDA_RawMetricsConfig_Create_Params metricsConfigOptions = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE};
#endif
    metricsConfigOptions.activityKind = NVPA_ACTIVITY_KIND_PROFILER;
    metricsConfigOptions.pChipName = chipName;

    NVPA_RawMetricsConfig *pRawMetricsConfig;
#if CUDART_VERSION < 11040
    NVPW_CALL(NVPA_RawMetricsConfig_Create(&metricsConfigOptions, &pRawMetricsConfig));
#else
    NVPW_CALL(NVPW_CUDA_RawMetricsConfig_Create(&metricsConfigOptions));
    pRawMetricsConfig = metricsConfigOptions.pRawMetricsConfig;
#endif

    NVPW_RawMetricsConfig_Destroy_Params rawMetricsConfigDestroyParams = {
        NVPW_RawMetricsConfig_Destroy_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    OnExit exit1([&]() { NVPW_RawMetricsConfig_Destroy(&rawMetricsConfigDestroyParams); });

    NVPW_RawMetricsConfig_BeginPassGroup_Params beginPassGroupParams = {
        NVPW_RawMetricsConfig_BeginPassGroup_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_BeginPassGroup(&beginPassGroupParams));

    NVPW_RawMetricsConfig_AddMetrics_Params addMetricsParams = {
        NVPW_RawMetricsConfig_AddMetrics_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
        .pRawMetricRequests = &rawMetricRequests[0],
        .numMetricRequests = rawMetricRequests.size(),
    };
    NVPW_CALL(NVPW_RawMetricsConfig_AddMetrics(&addMetricsParams));

    NVPW_RawMetricsConfig_EndPassGroup_Params endPassGroupParams = {
        NVPW_RawMetricsConfig_EndPassGroup_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_EndPassGroup(&endPassGroupParams));

    NVPW_RawMetricsConfig_GenerateConfigImage_Params generateConfigImageParams = {
        NVPW_RawMetricsConfig_GenerateConfigImage_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_GenerateConfigImage(&generateConfigImageParams));

    NVPW_RawMetricsConfig_GetConfigImage_Params getConfigImageParams = {
        NVPW_RawMetricsConfig_GetConfigImage_Params_STRUCT_SIZE,
        .pRawMetricsConfig = pRawMetricsConfig,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));

    configImage.resize(getConfigImageParams.bytesCopied);

    getConfigImageParams.bytesAllocated = configImage.size();
    getConfigImageParams.pBuffer = configImage.data();
    NVPW_CALL(NVPW_RawMetricsConfig_GetConfigImage(&getConfigImageParams));
}

void GetCounterDataPrefixImage(const char *chipName, std::vector<NVPA_RawMetricRequest> &rawMetricRequests,
                               std::vector<uint8_t> &counterDataImagePrefix) {
    NVPW_CounterDataBuilder_Create_Params counterDataBuilderCreateParams = {
        NVPW_CounterDataBuilder_Create_Params_STRUCT_SIZE,
        .pChipName = chipName,
    };
    NVPW_CALL(NVPW_CounterDataBuilder_Create(&counterDataBuilderCreateParams));

    NVPW_CounterDataBuilder_Destroy_Params counterDataBuilderDestroyParams = {
        NVPW_CounterDataBuilder_Destroy_Params_STRUCT_SIZE,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
    };
    OnExit exit0([&]() { NVPW_CounterDataBuilder_Destroy(&counterDataBuilderDestroyParams); });

    NVPW_CounterDataBuilder_AddMetrics_Params addMetricsParams = {
        NVPW_CounterDataBuilder_AddMetrics_Params_STRUCT_SIZE,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .pRawMetricRequests = &rawMetricRequests[0],
        .numMetricRequests = rawMetricRequests.size(),
    };
    NVPW_CALL(NVPW_CounterDataBuilder_AddMetrics(&addMetricsParams));

    size_t counterDataPrefixSize = 0;
    NVPW_CounterDataBuilder_GetCounterDataPrefix_Params getCounterDataPrefixParams = {
        NVPW_CounterDataBuilder_GetCounterDataPrefix_Params_STRUCT_SIZE,
        .pCounterDataBuilder = counterDataBuilderCreateParams.pCounterDataBuilder,
        .bytesAllocated = 0,
        .pBuffer = NULL,
    };
    NVPW_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));

    counterDataImagePrefix.resize(getCounterDataPrefixParams.bytesCopied);

    getCounterDataPrefixParams.bytesAllocated = counterDataImagePrefix.size();
    getCounterDataPrefixParams.pBuffer = counterDataImagePrefix.data();
    NVPW_CALL(NVPW_CounterDataBuilder_GetCounterDataPrefix(&getCounterDataPrefixParams));
}

bool GetImages(const char *chipName, std::vector<containers::string> metricNames, std::vector<uint8_t> &configImage,
               std::vector<uint8_t> &counterDataImagePrefix) {
    NVPW_CUDA_MetricsContext_Create_Params metricsContextCreateParams = {
        NVPW_CUDA_MetricsContext_Create_Params_STRUCT_SIZE,
        .pChipName = chipName,
    };
    NVPW_CALL(NVPW_CUDA_MetricsContext_Create(&metricsContextCreateParams));

    NVPW_MetricsContext_Destroy_Params metricsContextDestroyParams = {
        NVPW_MetricsContext_Destroy_Params_STRUCT_SIZE,
        .pMetricsContext = metricsContextCreateParams.pMetricsContext,
    };
    OnExit exit0([&]() { NVPW_MetricsContext_Destroy(&metricsContextDestroyParams); });

    std::vector<NVPA_RawMetricRequest> rawMetricRequests;
    std::vector<containers::string> temp;

    GetRawMetricRequests(metricsContextCreateParams.pMetricsContext, metricNames, rawMetricRequests, temp);
    GetConfigImage(chipName, rawMetricRequests, configImage);
    GetCounterDataPrefixImage(chipName, rawMetricRequests, counterDataImagePrefix);
    return true;
}

bool CreateCounterDataImage(std::vector<uint8_t> &counterDataImage, std::vector<uint8_t> &counterDataScratchBuffer,
                            std::vector<uint8_t> &counterDataImagePrefix) {
    uint32_t numRanges = GLOBALS.gpu.NUM_HWC_RANGES;
    CUpti_Profiler_CounterDataImageOptions counterDataImageOptions = {
        CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pCounterDataPrefix = counterDataImagePrefix.data(),
        .counterDataPrefixSize = counterDataImagePrefix.size(),
        .maxNumRanges = numRanges,
        .maxNumRangeTreeNodes = numRanges,
        .maxRangeNameLength = 64,
    };

    CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
        CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &counterDataImageOptions,
    };

    CUPTI_CALL(cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

    counterDataImage.resize(calculateSizeParams.counterDataImageSize);

    CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
        CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
        .sizeofCounterDataImageOptions = CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
        .pOptions = &counterDataImageOptions,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = counterDataImage.data(),
    };
    CUPTI_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

    CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params scratchBufferSizeParams = {
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,
        .pCounterDataImage = initializeParams.pCounterDataImage,
    };
    CUPTI_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(&scratchBufferSizeParams));

    counterDataScratchBuffer.resize(scratchBufferSizeParams.counterDataScratchBufferSize);

    CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params initScratchBufferParams = {
        CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
        .counterDataImageSize = calculateSizeParams.counterDataImageSize,

        .pCounterDataImage = initializeParams.pCounterDataImage,
        .counterDataScratchBufferSize = scratchBufferSizeParams.counterDataScratchBufferSize,
        .pCounterDataScratchBuffer = counterDataScratchBuffer.data(),
    };
    CUPTI_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(&initScratchBufferParams));

    return true;
}
}