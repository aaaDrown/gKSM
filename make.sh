cd /usr/lib/x86_64-linux-gnu
cp libcuda.so.1 libcuda-native.so
patchelf --set-soname libcuda-native.so libcuda-native.so

cd ~/dedup
nvcc -Xcompiler -fPIC -shared -o ./build/libcuda.so.1 \
     ./libcudahook.cpp ./tools.cu ./ipc.cpp \
     -ldl -lcrypto -lcuda-native -lcudart \
     -L/usr/lib/x86_64-linux-gnu/ -w

patchelf --set-soname libcuda.so.1 ./build/libcuda.so.1