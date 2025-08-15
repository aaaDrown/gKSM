# g++ -shared -fPIC -o ./build/libcuda.so ./libcudahook.cpp ./tools.cpp ./ipc.cpp     -ldl -lcrypto -lcuda-native     -L/usr/lib/x86_64-linux-gnu/    -w
# patchelf --set-soname libcuda.so ./build/libcuda.so
# patchelf --set-soname libcuda-native.so libcuda-native.so
# g++ -shared -fPIC -o ./build/libcuda.so.1 ./libcudahook.cpp ./tools.cpp ./ipc.cpp     -ldl -lcrypto -lcuda-native     -L/usr/lib/x86_64-linux-gnu/    -w
# patchelf --set-soname libcuda.so.1 ./build/libcuda.so.1
# rm /dev/shm/cuda_dedup_shared

# nvcc -Xcompiler -fPIC -c -o tools.o tools.cu 
# g++ -shared -fPIC -o ./build/libcuda.so.1 ./libcudahook.cpp ./tools.o ./ipc.cpp  -ldl -lcrypto -lcuda-native -L/usr/lib/x86_64-linux-gnu/  -w

nvcc -Xcompiler -fPIC -shared -o ./build/libcuda.so.1 \
     ./libcudahook.cpp ./tools.cu ./ipc.cpp \
     -ldl -lcrypto -lcuda-native -lcudart \
     -L/usr/lib/x86_64-linux-gnu/ -w

patchelf --set-soname libcuda.so.1 ./build/libcuda.so.1