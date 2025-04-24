{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=release-24.11";
    flake-utils.url = "github:numtide/flake-utils";
    formosa-llvm.url = "git+ssh://git@git.caslab.ee.ncku.edu.tw/formosa-gpgpu/formosa-llvm.git";
  };

  outputs = { self, nixpkgs, flake-utils, formosa-llvm }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        defaultPackage = flake: flake.packages."${system}".default;

        formosa-pocl = pkgs.callPackage ./package.nix {
          formosa-llvm = defaultPackage formosa-llvm;
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
