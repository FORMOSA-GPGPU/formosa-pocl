{ stdenv
, opencl-headers
, opencl-clhpp
, openssh
, git
, cmake
, ninja
, python3
, zlib
, hwloc
, ocl-icd
, clinfo
, pkg-config
, formosa-llvm
}:

stdenv.mkDerivation {
  name = "formosa-pocl";
  src = ./.;

  cmakeFlags = [
    "-D WITH_LLVM_CONFIG=${formosa-llvm}/bin/llvm-config"
    "-D LLC_TRIPLE=x86_64-unknown-linux-gnu"
    "-D ENABLE_ICD=ON"
    "-D ENABLE_FORMOSA=ON"
    "-D ENABLE_CUDA=OFF"
    "-D ENABLE_TCE=OFF"
    "-D ENABLE_HSA=OFF"
    "-D ENABLE_VULKAN=OFF"
    "-D ENABLE_LEVEL0=OFF"
    "-D CMAKE_INSTALL_PREFIX=${placeholder "out"}"
    "-D ENABLE_LATEST_CXX_STD=ON"
    "-D ENABLE_LIBLLVMOPENCL=ON"
    "-D ENABLE_HOST_CPU_DEVICES=OFF"
  ];

  AGENT_SOCKET_PATH = "/tmp/casvp-ipc";
  OCL_ICD_VENDORS = "${placeholder "out"}/etc/OpenCL/vendors";

  nativeBuildInputs = [
    openssh
    git
    cmake
    ninja
    python3
  ];

  buildInputs = [
    zlib
    opencl-headers
    opencl-clhpp
    ocl-icd
    pkg-config
    hwloc
    formosa-llvm
    clinfo
  ];
}
