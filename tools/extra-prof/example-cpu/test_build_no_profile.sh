export EXTRA_PROF_EVENT_TRACE=OFF
export EXTRA_PROF_DEBUG_BUILD=ON
export EXTRA_PROF_DEBUG_SANITIZE=OFF 

if [ ! -z ${ADD_CUDA_PATH+x} ]; then
export PATH=$PATH:$ADD_CUDA_PATH
echo $PATH
fi

rm library.o liblibrary.so lib_extra_prof.so test_exe
g++ -c -fPIC -o library.o library.cpp
g++ -shared -o liblibrary.so library.o 
g++ ParallelKernelsTest.cpp -g -fopenmp -o test_exe_no_profile -L. -llibrary -ldl 
