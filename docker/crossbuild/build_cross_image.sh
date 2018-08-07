#!/bin/bash

# Build Docker Image
if [ "$1" = "aarch64" ]
then
    # Once we support for targets, change this to the appropriate Docker image
    AUTOWARE_DOCKER_ARCH=arm64v8
    AUTOWARE_TARGET_ARCH=aarch64
    AUTOWARE_TARGET_PLATFORM=$1

    echo "Using ${AUTOWARE_TARGET_PLATFORM} as the target architecture"
    # Register QEMU as a handler for non-x86 targets
    docker run --rm --privileged multiarch/qemu-user-static:register

    # Build Docker Image
    docker build \
        --build-arg AUTOWARE_DOCKER_ARCH=${AUTOWARE_DOCKER_ARCH} \
        --build-arg AUTOWARE_TARGET_ARCH=${AUTOWARE_TARGET_ARCH} \
        --build-arg AUTOWARE_TARGET_PLATFORM=${AUTOWARE_TARGET_PLATFORM} \
        -t autoware/crossbuild:${AUTOWARE_TARGET_PLATFORM}-kinetic-1.0.0 \
        -f Dockerfile.kinetic-crossbuild .

    # Deregister QEMU as a handler for non-x86 targets
    docker run --rm --privileged multiarch/qemu-user-static:register --reset
else
    echo "Select target platform: aarch64"
fi
