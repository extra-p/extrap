Extra-Profiler
==============

The Extra-Profiler profiles both CPU and GPU execution jointly. It creates profiles with one call tree that encompasses both executions.

Instrumentation
---------------
Before you can start the measurement, please, compile your application using the compiler wrappers
in this folder. The compiler wrappers are mainly designed for GCC compatible compilers.

You can directly replace the call to your compiler with the corresponding wrapper:

```sh
nvcc [compiler arguments] # original command
./nvcc-wrapper.sh [compiler arguments] # wrapped command
```

### Other compilers

If you want to use a different compiler you can make your own wrapper by using one of the existing ones as a starting
point.

### Build systems

If you need to use the wrapper as a compiler replacement for a build tool which prohibits that the compiler value
includes any flags (e.g., CMake). You can temporarily disable the wrapper by setting `EXTRA_PROF_WRAPPER=off`.

### Options

You can set the following options as environment variables during compilation.

| Option                                                    | Description                                                                                                                                                              | 
|-----------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EXTRA_PROF_WRAPPER`=`on`(default),`off` `                | Enables the compiler wrapper                                                                                                                                             |
| `EXTRA_PROF_GPU`=`on`(default),`off`                      | Enables GPU measurements                                                                                                                                                 |
| `EXTRA_PROF_EVENT_TRACE`=`on`,`gpu_only`,`off`(default)   | Enables recording of event traces in Trace Event Format                                                                                                                  |
| `EXTRA_PROF_ENERGY`=`on`,`off`(default)                   | Enables energy measurements                                                                                                                                              |
| `EXTRA_PROF_ADVANCED_INSTRUMENTATION`=`on`(default),`off` | Improves instrumentation by not instrumenting standard library functions. *Only works with compilers that support the `-finstrument-functions-exclude-file-list` option* |
| `EXTRA_PROF_EXCLUDE_FILES`= *FILE*, *FILE*, ...           | Excludes the listed files from the instrumentation. *Only works with compilers that support the `-finstrument-functions-exclude-file-list` option*                       |
| `EXTRA_PROF_EXCLUDE_FUNCTIONS`= *SYMBOL*, *SYMBOL*, ...   | Excludes the listed functions from the instrumentation. *Only works with compilers that support the `-finstrument-functions-exclude-function-list` option*               |
| `EXTRA_PROF_SCOREP_INSTRUMENTATION`= *PATH*               | Set the path to the Score-P compiler plugin to use it for the instrumentation of the CPU code. *Disables all other CPU instrumentation settings.*                        |

The following advanced options can be used when creating new compiler wrappers and during development and debugging.

| Option                                                    | Description                                                                                                                            |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `EXTRA_PROF_COMPILER`=*COMPILER COMMAND*                  | Sets the compiler the Extra-Prof wrapper emulates                                                                                      |
| `EXTRA_PROF_INTERNAL_COMPILER`=*COMPILER COMMAND*         | Sets the compiler that is used by the Extra-Prof wrapper to compile the Extra-Prof runtime components                                  |
| `EXTRA_PROF_COMPILER_OPTION_REDIRECT`= *REDIRECTION FLAG* | Sets the prefix to add before any flag supposed for the compiler in the compilation process (e.g., `-Xcompiler` for nvcc)              |
| `EXTRA_PROF_DEBUG`=`on`,`off`(default)                    | Enables debugging output of the profiling. *This might cause performance issues when profiling.*                                       |
| `EXTRA_PROF_DEBUG_BUILD`=`on`,`off`(default)              | Sets the compiler flags to allow debugging of the Extra-Prof runtime components. *This might cause performance issues when profiling.* |
| `EXTRA_PROF_DEBUG_INSTRUMENTATION`=`on`,`off`(default)    | Adds additional checks to the instrumentation code, helpful during development. *This might cause performance issues when profiling.*  |                                            
| `EXTRA_PROF_DEBUG_SANITIZE`=`on`,`off`(default)           | Enables address sanitation for the runtime components and the profiled application. *This might cause issues when profiling.*          |                                                          |

Measurement
-----------
Similar to measurements performed with Score-P, the measurements with Extra-Prof must be stored in a folder
hierarchy, so that Extra-P can import them. You can read more about the folder structure in
the [file formats documentation](file-formats.md#nsight-systems-with-extra-prof-data-file-format).

You should repeat every measurement at least five times and measure at least five configurations per parameter.

### Performing a measurement

Like Score-P the executable compiled with the Extra-Prof wrappers, performs the profiling on its own, so the command
typically looks like:

```sh
srun -n 4 <application> <application arguments>
```

After you have executed the command you will find a new folder that contains the results for the measurement.
You can customize the name of the folder, that contains the result by setting the `EXTRA_PROF_EXPERIMENT_DIRECTORY`
environment variable. *Please
note, that the folder will be overridden, when a new measurement is performed with the same results directory.*
You can additionally specify the callpath-depth, by setting the `EXTRA_PROF_MAX_DEPTH` environment variable.
This might be when the reports get to big.

### Options

| Option                                                 | Description                                                                                                                                                                                                                                                                                   | 
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EXTRA_PROF_EXPERIMENT_DIRECTORY`                      | Sets the name of the folder, which contains the result of the measurement.                                                                                                                                                                                                                    |                                                                                                                                                                                                     
| `EXTRA_PROF_MAX_DEPTH`                                 | Specifies the maximum callpath depth, that is still recorded. *Does not affect GPU measurements.*                                                                                                                                                                                             |                                                                                                                                                                                
| `EXTRA_PROF_CUPTI_BUFFER_SIZE` (default: 1MB)          | Specifies the size (in Bytes) of the buffer reserved for collecting GPU activities via CUPTI                                                                                                                                                                                                  |
| `EXTRA_PROF_HANDLE_NO_GPU`= `error`(default),`warning` | Specifies how to handle the case when no GPU is found during execution                                                                                                                                                                                                                        |
| `EXTRA_PROF_GPU_METRICS`                               | Specifies the comma-separated list of GPU HW counters to profile (see [Nsight Metrics Structure](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)) *Selecting one or more metrics serializes the kernel execution, which may impact the CPU measurements.* |
| `EXTRA_PROF_GPU_HWC_RANGES` = 256 (default)            | Number of ranges (kernel launches) that are profiled before the CUPTI profiling session is restarted. *Large numbers can drastically increase memory overhead.*                                                                                                                               |
| `EXTRA_PROF_GPU_HWC_REPLAY` = `none`,`kernel`(default) | Selects the kernel replay strategy                                                                                                                                                                                                                                                            |
| `EXTRA_PROF_CPU_ENERGY_COUNTER_PATH`                   | Sets a path to a file that contains the current energy usage of the CPU                                                                                                                                                                                                                       |
| `EXTRA_PROF_GPU_ENERGY_COUNTER_PATH`                   | Sets a path to a file that contains the current energy usage of the GPU                                                                                                                                                                                                                       |        

Viewing Profiles
----------------

The Extra-Profiler comes with a small utility to view the contents of the profiles. It requires Python 3.8 or newer and the Python package *msgpack* to read the Extra-Profiler files. You can install that via pip:

```sh
pip install msgpack
```

You can use the viewer on the command line to have a look at the collected metrics of an Extra-Profiler file:

```sh
python profile-viewer.py <FILE.extra-prof.msgpack>
```