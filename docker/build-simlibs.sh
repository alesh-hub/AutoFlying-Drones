#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/ue/NoobStage1/Plugins/ProjectAirSim"
UE="/opt/Linux_Unreal_Engine_5.2.0"

docker run --rm -it \
  --user $(id -u):$(id -g) \
  --network host \
  -v "$SRC":/src \
  -v "$UE":/ue:ro \
  -e DEBIAN_FRONTEND=noninteractive \
  ubuntu:20.04 /bin/bash -lc '
    apt-get update &&
    apt-get install -y build-essential ninja-build git curl python3 cmake libxi-dev libxcursor-dev libxrandr-dev libxinerama-dev \
                       libgl1-mesa-dev libglu1-mesa-dev libasound2-dev libpulse-dev libudev-dev libdbus-1-dev rsync &&
    export UE_ROOT=/ue &&
    export UE5_ROOT=/ue &&
    export PATH="$UE_ROOT/Engine/Extras/ThirdPartyNotUE/SDKs/HostLinux/Linux_x64/v21_clang-15.0.1-centos7/x86_64-unknown-linux-gnu/bin:$PATH" &&
    cd /src &&
    cmake -S . -B build/linux64/Release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="$PWD/unreal-linux-toolchain.cmake" &&
    cmake --build build/linux64/Release -j"$(nproc)" &&
    make -f build_linux.mk package_simlibs
  '
