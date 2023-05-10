
#pragma once
#include "../common_types.h"
#include "../globals.h"
#include "gpu_instrumentation.h"

#include "gpu_hwc_images.h"
#include <mutex>
namespace extra_prof::gpu::hwc {

void start_profiling_phase() {
    int currentDeviceIdx;
    GPU_CALL(cudaGetDevice(&currentDeviceIdx));
    CUdevice currentDevice;
    CUresult resul1 = cuDeviceGet(&currentDevice, currentDeviceIdx);
    CUcontext currentContext;
    CUresult result = cuDevicePrimaryCtxRetain(&currentContext, currentDevice);
    // std::cout << "Res: " << result << " Context: " << currentContext << std::endl;

    size_t numRanges = GLOBALS.gpu.NUM_HWC_RANGES;
    CUpti_Profiler_BeginSession_Params beginSessionParams = {
        CUpti_Profiler_BeginSession_Params_STRUCT_SIZE,
        .ctx = currentContext,
        .counterDataImageSize = GLOBALS.gpu.counterDataImage.size(),
        .pCounterDataImage = GLOBALS.gpu.counterDataImage.data(),
        .counterDataScratchBufferSize = GLOBALS.gpu.counterDataScratchBuffer.size(),
        .pCounterDataScratchBuffer = GLOBALS.gpu.counterDataScratchBuffer.data(),
        .range = CUPTI_AutoRange,
        .replayMode = CUPTI_KernelReplay,
        .maxRangesPerPass = numRanges,
        .maxLaunchesPerPass = numRanges,
    };
    CUPTI_CALL(cuptiProfilerBeginSession(&beginSessionParams));

    CUpti_Profiler_SetConfig_Params setConfigParams = {
        CUpti_Profiler_SetConfig_Params_STRUCT_SIZE,

        .ctx = currentContext,
        .pConfig = GLOBALS.gpu.configImage.data(),
        .configSize = GLOBALS.gpu.configImage.size(),
        .passIndex = 0,
    };
    CUPTI_CALL(cuptiProfilerSetConfig(&setConfigParams));

    CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
        CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE,
        .ctx = currentContext,
    };
    CUPTI_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
}

void end_profiling_phase() {
    CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
        CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

    CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
    CUpti_Profiler_EndSession_Params endSessionParams = {CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerEndSession(&endSessionParams));
}

std::string GetHwUnit(const std::string &metricName) { return metricName.substr(0, metricName.find("__", 0)); }

struct MetricNameValue {
    std::string metricName;
    int numRanges;
    // <rangeName , metricValue> pair
    std::vector<std::pair<std::string, double>> rangeNameMetricValueMap;
};

void postprocess_counter_data() {
    const char *chipName = GLOBALS.gpu.chipName.c_str();
    const std::vector<uint8_t> &counterDataImage = GLOBALS.gpu.counterDataImage;
    const std::vector<std::string> &metricNames = GLOBALS.gpu.metricNames;

    if (!counterDataImage.size()) {
        throw std::runtime_error("Counter Data Image is empty!");
    }

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

    NVPW_CounterData_GetNumRanges_Params getNumRangesParams = {
        NVPW_CounterData_GetNumRanges_Params_STRUCT_SIZE,
        .pCounterDataImage = counterDataImage.data(),
    };
    NVPW_CALL(NVPW_CounterData_GetNumRanges(&getNumRangesParams));

    std::vector<const char *> metricNamePtrs;
    for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
        metricNamePtrs.push_back(metricNames[metricIndex].c_str());
    }
    std::vector<double> gpuValues;
    gpuValues.resize(metricNames.size());
    for (size_t rangeIndex = 0; rangeIndex < getNumRangesParams.numRanges; ++rangeIndex) {
        std::vector<const char *> descriptionPtrs;

        NVPW_Profiler_CounterData_GetRangeDescriptions_Params getRangeDescParams = {
            NVPW_Profiler_CounterData_GetRangeDescriptions_Params_STRUCT_SIZE,
            .pCounterDataImage = counterDataImage.data(),
            .rangeIndex = rangeIndex,
        };
        NVPW_CALL(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));
        descriptionPtrs.resize(getRangeDescParams.numDescriptions);

        getRangeDescParams.ppDescriptions = descriptionPtrs.data();
        NVPW_CALL(NVPW_Profiler_CounterData_GetRangeDescriptions(&getRangeDescParams));

        std::string rangeName;
        for (size_t descriptionIndex = 0; descriptionIndex < getRangeDescParams.numDescriptions; ++descriptionIndex) {
            if (descriptionIndex) {
                rangeName += "/";
            }
            rangeName += descriptionPtrs[descriptionIndex];
        }

        NVPW_MetricsContext_SetCounterData_Params setCounterDataParams = {
            NVPW_MetricsContext_SetCounterData_Params_STRUCT_SIZE,
            .pMetricsContext = metricsContextCreateParams.pMetricsContext,
            .pCounterDataImage = counterDataImage.data(),
            .rangeIndex = rangeIndex,
            .isolated = true,
        };
        NVPW_MetricsContext_SetCounterData(&setCounterDataParams);

        NVPW_MetricsContext_EvaluateToGpuValues_Params evalToGpuParams = {
            NVPW_MetricsContext_EvaluateToGpuValues_Params_STRUCT_SIZE,
            .pMetricsContext = metricsContextCreateParams.pMetricsContext,
            .numMetrics = metricNamePtrs.size(),
            .ppMetricNames = metricNamePtrs.data(),
            .pMetricValues = gpuValues.data(),
        };
        NVPW_MetricsContext_EvaluateToGpuValues(&evalToGpuParams);

        auto correlationId = GLOBALS.gpu.rangeToCorrelationId[rangeIndex];
        auto *correlationData = GLOBALS.gpu.callpath_correlation.try_get(correlationId);
        if (correlationData == nullptr) {
            throw std::runtime_error(std::string("Correlation data not found for id ") + std::to_string(correlationId));
        }
        auto &gpu_metrics = correlationData->node->gpu_metrics;

        gpu_metrics.resize(metricNamePtrs.size());

        for (size_t metricIndex = 0; metricIndex < metricNames.size(); ++metricIndex) {
            gpu_metrics[metricIndex] += gpuValues[metricIndex];
        }
    }
}

void onKernelLaunch(const CUpti_CallbackData *cbdata) {
    std::lock_guard lg(GLOBALS.gpu.kernelLaunchMutex);

    if (GLOBALS.gpu.rangeCounter >= GLOBALS.gpu.NUM_HWC_RANGES) {
        end_profiling_phase();
        postprocess_counter_data();
        start_profiling_phase();
        GLOBALS.gpu.rangeCounter = 0;
    }
    GLOBALS.gpu.rangeToCorrelationId[GLOBALS.gpu.rangeCounter] = cbdata->correlationId;
    GLOBALS.gpu.rangeCounter++;
    GLOBALS.gpu.totalRangeCounter++;
}
void init() {

    GLOBALS.gpu.onKernelLaunch = onKernelLaunch;

    const char *EXTRA_PROF_GPU_HWC_RANGES = getenv("EXTRA_PROF_GPU_HWC_RANGES");
    if (EXTRA_PROF_GPU_HWC_RANGES != nullptr) {
        char *end;
        GLOBALS.gpu.NUM_HWC_RANGES = std::strtoul(EXTRA_PROF_GPU_HWC_RANGES, &end, 10);
    }
    GLOBALS.gpu.rangeToCorrelationId.resize(GLOBALS.gpu.NUM_HWC_RANGES);

    CUpti_Profiler_Initialize_Params profilerInitializeParams = {CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerInitialize(&profilerInitializeParams));

    int currentDeviceIdx;
    GPU_CALL(cudaGetDevice(&currentDeviceIdx));

    /* Get chip name for the cuda  device */
    CUpti_Device_GetChipName_Params getChipNameParams = {
        CUpti_Device_GetChipName_Params_STRUCT_SIZE,
        .deviceIndex = (size_t)currentDeviceIdx,
    };
    CUPTI_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    GLOBALS.gpu.chipName = getChipNameParams.pChipName;

    NVPW_InitializeHost_Params initializeHostParams = {NVPW_InitializeHost_Params_STRUCT_SIZE};
    NVPW_CALL(NVPW_InitializeHost(&initializeHostParams));

    const char *EXTRA_PROF_GPU_METRICS = getenv("EXTRA_PROF_GPU_METRICS");
    if (EXTRA_PROF_GPU_METRICS != nullptr) {
        std::string metrics_string(EXTRA_PROF_GPU_METRICS);
        metrics_string += ',';

        for (size_t start = 0, end = metrics_string.find(',', start); end != std::string::npos;
             start = end + 1, end = metrics_string.find(',', start)) {
            GLOBALS.gpu.metricNames.emplace_back(metrics_string.substr(start, end - start));
        }
    }

    if (!GLOBALS.gpu.metricNames.size()) {
        std::cerr << "EXTRA PROF: Missing GPU metrics";
        exit(-1);
    }
    if (!GetImages(GLOBALS.gpu.chipName.c_str(), GLOBALS.gpu.metricNames, GLOBALS.gpu.configImage,
                   GLOBALS.gpu.counterDataImagePrefix)) {
        std::cout << "Failed to create configImage" << std::endl;
        exit(-1);
    }
    if (!CreateCounterDataImage(GLOBALS.gpu.counterDataImage, GLOBALS.gpu.counterDataScratchBuffer,
                                GLOBALS.gpu.counterDataImagePrefix)) {
        std::cout << "Failed to create counterDataImage" << std::endl;
        exit(-1);
    }

    std::cerr << "EXTRA PROF: GPU counter image size: "
              << GLOBALS.gpu.counterDataImage.size() + GLOBALS.gpu.counterDataScratchBuffer.size() +
                     GLOBALS.gpu.counterDataImagePrefix.size() + GLOBALS.gpu.configImage.size()
              << std::endl;

    start_profiling_phase();
}
void finalize() {
    end_profiling_phase();
    CUresult result = cuDevicePrimaryCtxRelease(0);

    CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
    CUPTI_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}
}
