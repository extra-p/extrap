export PATH=$PATH:/usr/local/cuda/bin
export EXTRA_PROF_EVENT_TRACE=ON
export EXTRA_PROF_DEBUG_BUILD=ON
export EXTRA_PROF_DEBUG_SANITIZE=OFF 
export EXTRA_PROF_GPU_HWC=ON

rm library.o liblibrary.so lib_extra_prof.so test_exe
/home/alexandergeiss/extra_prof/nvcc-wrapper.sh -c -Xcompiler -fPIC -o library.o library.cpp
/home/alexandergeiss/extra_prof/nvcc-wrapper.sh -shared -o liblibrary.so library.o 
/home/alexandergeiss/extra_prof/nvcc-wrapper.sh kernels.cu copy_test.cu max_parallel_test.cu ParallelKernelsTest.cpp -g -Xcompiler -fopenmp -o test_exe -L. -L /usr/local/cuda/lib64 -I /usr/local/cuda/include -lcuda -llibrary -ldl 

# g++ --shared -o threadwrap.so ../extra_prof/globals.cpp wraplibrary.cpp -fPIC -I../msgpack/include -I /usr/local/cuda/include