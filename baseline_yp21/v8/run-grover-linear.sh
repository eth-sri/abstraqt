#!/bin/bash

echo
echo "grover150-zero-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover150-zero-linear.q

echo
echo "grover32-plus-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover32-plus-linear.q

echo
echo "grover150-plus-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover150-plus-linear.q

echo
echo "grover128-plus-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover128-plus-linear.q

echo
echo "grover64-plus-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover64-plus-linear.q

echo
echo "grover128-zero-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover128-zero-linear.q

echo
echo "grover64-zero-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover64-zero-linear.q

echo
echo "grover32-zero-linear; k=5"
java Main static 5 timed gate ex 3 < benchmarks/single-it-grover32-zero-linear.q

