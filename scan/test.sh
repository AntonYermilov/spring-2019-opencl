#!/bin/bash

mkdir -p build
cd build
cmake ..
make
cd ..

arg0=(15  16  17  1023  1024  1025  65535  65536  65537  98304  98305  1048576  1048575)
arg0=("${arg0[@]}")

for ((i = 0; i != 13; i++)) do
    echo ${arg0[${i}]}
    ./build/generator ${arg0[${i}]} > input.txt
    ./build/scan
    ./build/checker
done

rm -rf build
rm "input.txt" "output.txt"
