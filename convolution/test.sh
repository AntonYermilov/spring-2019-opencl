#!/bin/bash

mkdir -p build
cd build
cmake ..
make
cd ..

arg0=(1024 1024 1 31 1023)
arg1=(3 9 9 9 9)
arg0=("${arg0[@]}")
arg1=("${arg1[@]}")

for ((i = 0; i != 5; i++)) do
    ./build/generator ${arg0[${i}]} ${arg1[${i}]} > input.txt
    ./build/convolution
    ./build/checker ${arg0[${i}]} ${arg1[${i}]} < output.txt
done

rm -rf build
rm "input.txt" "output.txt"
