# PoCL with FORMOSA device driver

## Building PoCL

### Option A: direnv (recommended)

If this `pocl` checkout sits next to a `formosa` clone:

```text
<formosa-gpgpu>/
  formosa/
  pocl/        # you are here
```

then:

```bash
# once per clone
direnv allow

# entering pocl/ loads formosa#formosa-pocl automatically
cmake -B build -G Ninja $(echo $cmakeFlags)
cmake --build build --target install
```

`direnv` only auto-discovers **sibling** `../formosa` (same parent directory as `pocl`).

If formosa is elsewhere, set an absolute path once:

```bash
export FORMOSA_FLAKE=/absolute/path/to/formosa
direnv allow   # re-enter pocl/ after setting
```

Requires [direnv](https://direnv.net/) with [nix-direnv](https://github.com/nix-community/nix-direnv), same as the formosa monorepo.

### Option B: manual nix develop

```bash
nix develop /path/to/formosa#formosa-pocl
cmake -B build -G Ninja $(echo $cmakeFlags)
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
