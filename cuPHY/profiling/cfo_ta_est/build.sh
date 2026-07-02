#!/bin/bash
set -e

if [ "$1" == "clean" ]; then
    rm -rf build
    exit 0
fi

mkdir -p build
cd build
cmake ..
make -j
