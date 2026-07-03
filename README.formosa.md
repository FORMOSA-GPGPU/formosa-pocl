# PoCL with FORMOSA device driver

## Building PoCL

1. Configure

Use the Nix dev shell provided by the [formosa](https://git.caslab.ee.ncku.edu.tw/formosa-gpgpu/formosa) project.

```bash
nix develop /path/to/formosa#formosa-pocl
cmake -B build -G Ninja `echo $cmakeFlags`
```

2. Build & Install
```bash
sudo cmake --build build --target install
```

## Examine the installation
1. Start the lv server
2. Export the socket path
```bash
export AGENT_SOCKET_PATH=/tmp/lv-ipc
```
3. Run clinfo to check if the FORMOSA device is available
```bash
clinfo -l
```

## Running PoCL with FORMOSA
1. Set the environment variable
```bash
export AGENT_SOCKET_PATH=<path-to-lv-ipc-socket>
export POCL_FORMOSA_CFLAGS=<kernel-compiler-flags>  # default: "-O2" (no PRI insertion); for PRI, use: -fsa-ics-first, -mllvm -fsa-ipdom-like, or -mllvm -fsa-post-topo
export POCL_FORMOSA_LDFLAGS=<kernel-linker-flags>  # default: "-fuse-ld=lld -nostartfiles"
```
2. Run the OpenCL program
```bash
./path/to/your/opencl/program
```
