export PATH=$PATH:/usr/local/cuda/bin
export EXTRA_PROF_EVENT_TRACE=OFF
export EXTRA_PROF_DEBUG_BUILD=ON
export EXTRA_PROF_DEBUG_SANITIZE=ON 

rm library.o liblibrary.so lib_extra_prof.so test_exe
../g++-wrapper.sh -c -fPIC -o library.o library.cpp
../g++-wrapper.sh -shared -o liblibrary.so library.o 
../g++-wrapper.sh ParallelKernelsTest.cpp -g -fopenmp -o test_exe -L. -llibrary -ldl 
