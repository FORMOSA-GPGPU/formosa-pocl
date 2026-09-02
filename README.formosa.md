# PoCL with FORMOSA device driver

## Building PoCL

### Option A: direnv (recommended)

```bash
cp .envrc.local.example .envrc.local
# Set the required FORMOSA_SOURCE_DIR, then:
direnv allow

# entering pocl/ loads formosa#formosa-pocl automatically
cmake -B build -G Ninja $(echo $cmakeFlags)
cmake --build build --target install
```

`FORMOSA_SOURCE_DIR` must point to a FORMOSA source tree containing `flake.nix`.
PoCL does not infer this path from the checkout layout.

PoCL links the installed `Formosa::formosa-hal.lv` CMake target. By default it
uses the FORMOSA SDK from the Nix shell. To use a locally installed FORMOSA SDK,
set `FORMOSA_INSTALL_PREFIX` in `.envrc.local`, then run `direnv reload`.

`FORMOSA_INSTALL_PREFIX` is an install prefix, not a source directory. Build and
install FORMOSA first; the prefix must contain
`lib/cmake/Formosa/FormosaConfig.cmake` or
`lib64/cmake/Formosa/FormosaConfig.cmake`.

When migrating from the old source-tree setup, configure PoCL in a clean build
directory.

Requires [direnv](https://direnv.net/) with [nix-direnv](https://github.com/nix-community/nix-direnv), same as the formosa monorepo.

### Option B: manual nix develop

```bash
nix develop /path/to/formosa#formosa-pocl
cmake -B build -G Ninja $(echo $cmakeFlags) \
  -D Formosa_DIR=/path/to/formosa-install/lib64/cmake/Formosa
cmake --build build --target install
```

## Examine the installation

1. Start the lv server (from the formosa monorepo)
2. Export the socket path (direnv defaults to `/tmp/formosa.sock`)
```bash
export AGENT_SOCKET_PATH=/tmp/formosa.sock
```
3. Point ICD at your local install if needed
```bash
export OCL_ICD_VENDORS=$CMAKE_INSTALL_PREFIX/etc/OpenCL/vendors/pocl.icd
```
4. Run clinfo to check if the FORMOSA device is available
```bash
clinfo -l
```

## Running PoCL with FORMOSA

1. Set the environment variable
```bash
export AGENT_SOCKET_PATH=<path-to-formosa.sock>
export POCL_FORMOSA_CFLAGS=<kernel-compiler-flags>  # default: "-O2" (no PRI insertion); for PRI, use: -fsa-ics-first or -mllvm -fsa-ipdom-like
export POCL_FORMOSA_LDFLAGS=<kernel-linker-flags>  # default: "-fuse-ld=lld -nostartfiles"
```
2. Run the OpenCL program
```bash
./path/to/your/opencl/program
```
