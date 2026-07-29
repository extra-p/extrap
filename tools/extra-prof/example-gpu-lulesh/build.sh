if [ ! -z ${ADD_CUDA_PATH+x} ]; then
export PATH=$PATH:$ADD_CUDA_PATH
echo $PATH
fi

../nvcc-wrapper.sh src/allocator.cu src/lulesh-comms-gpu.cu src/lulesh-comms.cu src/lulesh.cu -I src -o lulesh
