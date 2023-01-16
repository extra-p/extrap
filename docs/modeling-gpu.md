Measuring and Modeling GPU Activities
=====================================

Extra-P supports the modeling of GPU activities like kernels and data transfers by default.
If you want to use the extended functionality of integrating GPU measurements into the application calltree and wait
state and concurrency analysis, you must import a measurement with additional metadata. Currently, Extra-P requires you
to use Nvidia Nsight Systems together with a custom NVTX-based instrumenter to gather the required measurements'
metadata.


Instrumentation
---------------
Before you can start the measurement, please, compile your application using the compiler wrappers
in [/tools/extra-prof-nv](/tools/extra-prof-nv). The compiler wrappers are designed for GCC compatible compilers.

You can directly replace the call to your compiler:

```sh
nvcc [compiler arguments] # original command
./nvcc-wrapper.sh [compiler arguments] # wrapped command
```

### Other compilers

If you want to use a different compiler you can make your own wrapper by using `mpicc-wrapper` as a starting point.
Alternatively, you have to enable function instrumentation in your compiler and
include [/tools/extra-prof-nv/instrumentation.cpp](/tools/extra-prof-nv/instrumentation.cpp) into your build.

### Build systems

If you need to use the wrapper as a compiler replacement for a build tool which prohibits that the compiler value
includes any flags (e.g., CMake). You can temporarily disable the wrapper by setting `EXTRA_PROF_WRAPPER=off`.


Measurement
-----------
Similar to measurements performed with Score-P, the measurements with Nsight Systems must be stored in a folder
hierarchy, so that Extra-P can import them. You can read more about the folder structure in
the [file formats documentation](file-formats.md#nsight-systems-with-extra-prof-data-file-format).

You should repeat every measurement at least five times and measure at least five configurations per parameter.

### Performing a measurement

For measuring of GPU activity you need to have Nsight Systems installed. You can
use [/tools/extra-prof-nv/extra-prof.sh](/tools/extra-prof-nv/extra-prof.sh) to execute Nsight Systems with the correct
settings for the measurement.
*Please note, that `extra-prof.sh` is intended to be run in a Slurm environment so that it
can retrieve the `SLURM_PROCID` environment variable to ensure that each rank writes to a separate file.*
Typically, the measurement is started with a command similar to :

```sh
srun -n 4 ./extra-prof.sh [optional Nsight arguments] <application> <application arguments>
```

> ###### Hint
> If you pass arguments to Nsight Systems, please make sure that the output of sqlite files is still enabled.

After you have executed the command you will find a new folder that contains the results for the measurement.
You can customize the name of the folder, that contains the result by setting the `EXTRA_PROF_EXPERIMENT_DIRECTORY`
environment variable. For compatibility reasons the `SCOREP_EXPERIMENT_DIRECTORY` variable can be used instead. *Please
note, that the folder will be overridden, when a new measurement is performed with the same results directory.*
You can additionally specify the callpath-depth, by setting the `EXTRA_PROF_MAX_DEPTH` environment variable. 
This might be needed if your measurements crashes, because the reports get to big.


Modeling
--------

You can import the measurements into Extra-P by selecting the menu entry *Open set of Nsight files* in the *File* menu
or via the commandline by specifying `--nsight`. Extra-P will automatically create models for the call-paths of your
application and the GPU activities. The GPU activities will be placed inside the call-tree so that they appear as a
child of the calling function. Because GPU activities run concurrent to the CPU activities, Extra-P also creates models
for overlapping execution such as *GPU IDLE*, *EVENT_SYNCHRONIZE*, *WAIT*, and *OVERLAP*. Hereby, *OVERLAP* is special
as it describes at least two GPU activities running in parallel.

