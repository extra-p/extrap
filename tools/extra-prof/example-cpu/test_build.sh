export PATH=$PATH:/usr/local/cuda/bin
export EXTRA_PROF_EVENT_TRACE=OFF
export EXTRA_PROF_DEBUG_BUILD=ON
export EXTRA_PROF_DEBUG_SANITIZE=OFF 

rm library.o liblibrary.so lib_extra_prof.so test_exe
/home/alexandergeiss/extra_prof/g++-wrapper.sh -c -fPIC -o library.o library.cpp
/home/alexandergeiss/extra_prof/g++-wrapper.sh -shared -o liblibrary.so library.o 
/home/alexandergeiss/extra_prof/g++-wrapper.sh ParallelKernelsTest.cpp -g -fopenmp -o test_exe -L. -llibrary -ldl 

# g++ --shared -o threadwrap.so ../extra_prof/globals.cpp wraplibrary.cpp -fPIC -I../msgpack/include -I /usr/local/cuda/include