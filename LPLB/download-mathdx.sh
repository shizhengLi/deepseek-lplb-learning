#!/bin/bash

VERSION="25.06.0"

echo Downloading MathDx $VERSION

# Download and extract the appropriate version
if [[ ! -f "nvidia-mathdx-$VERSION.tar.gz" ]]; then
    wget "https://developer.download.nvidia.com/compute/cuSOLVERDx/redist/cuSOLVERDx/nvidia-mathdx-$VERSION.tar.gz"
else
    echo "Files already exist - skipping download"
fi

TARGET_DIR=lplb/resources/mathdx
echo Extracting nvidia-mathdx-$VERSION.tar.gz to $TARGET_DIR
tar -xf "nvidia-mathdx-$VERSION.tar.gz"
rm -rf $TARGET_DIR
mv "nvidia-mathdx-$VERSION/nvidia/mathdx/${VERSION%.*}" $TARGET_DIR
rm -rf "nvidia-mathdx-$VERSION"
