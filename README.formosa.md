# PoCL with FORMOSA device driver

## Building PoCL

1. Install the dependencies
```bash
sudo apt update
sudo apt install -y ocl-icd-libopencl1 pkg-config \
                    ocl-icd-dev ocl-icd-opencl-dev \
                    libhwloc-dev zlib1g zlib1g-dev \
                    clinfo dialog apt-utils libxml2-dev
```


2. Setup the toolchain paths
```bash
export FORMOSA_LLVM=<path-to-formosa-llvm>
export FORMOSA_CASVP=<path-to-casvp>
```

3. Configuration
```bash
cmake -B build -G Ninja \
      -D WITH_LLVM_CONFIG=${FORMOSA_LLVM}/bin/llvm-config \
      -D LLC_TRIPLE="x86_64-unknown-linux-gnu" \
      -D ENABLE_ICD=ON \
      -D ENABLE_FORMOSA=ON \
      -D ENABLE_CUDA=OFF \
      -D ENABLE_TCE=OFF \
      -D ENABLE_HSA=OFF \
      -D ENABLE_VULKAN=OFF \
      -D ENABLE_LEVEL0=OFF \
      -D CMAKE_INSTALL_PREFIX=/usr \
      -D ENABLE_LATEST_CXX_STD=ON \
      -D ENABLE_LIBLLVMOPENCL=ON \
      -D ENABLE_HOST_CPU_DEVICES=ON \
      -D CASVP_INSTALL_DIR=${FORMOSA_CASVP}
```

4. Build & Install
```bash
sudo cmake --build build --target install
```

## Examine the installation
1. Start the casvp server
2. Export the socket path
```bash
export AGENT_SOCKET_PATH=/tmp/casvp-ipc
```
3. Run clinfo to check if the FORMOSA device is available
```bash
clinfo -l
```
