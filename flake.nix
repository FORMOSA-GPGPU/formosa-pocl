{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    formosa-llvm.url = "git+ssh://git@git.caslab.ee.ncku.edu.tw/formosa-gpgpu/formosa-llvm.git?shallow=1";
    casvp.url = "git+ssh://git@git.caslab.ee.ncku.edu.tw/caslab-virtual-platform/casvp.git?submodules=1";
  };

  outputs = { self, nixpkgs, flake-utils, formosa-llvm, casvp }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        defaultPackage = flake: flake.packages."${system}".default;

        formosa-pocl = pkgs.callPackage ./package.nix {
          formosa-llvm = defaultPackage formosa-llvm;
          casvp = defaultPackage casvp;
        };
      in
      {
        packages = {
          inherit formosa-pocl;
          default = formosa-pocl;
        };
      }
    );
}
