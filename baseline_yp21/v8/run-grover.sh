#!/bin/bash

echo
echo "grover4-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover4-zero.q

echo
echo "grover4-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover4-plus.q

echo
echo "grover8-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover8-zero.q

echo
echo "grover8-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover8-plus.q

echo
echo "grover16-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover16-zero.q

echo
echo "grover16-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover16-plus.q

echo
echo "grover32-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover32-zero.q

echo
echo "grover32-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover32-plus.q

echo
echo "grover64-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover64-zero.q

echo
echo "grover64-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover64-plus.q

echo
echo "grover128-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover128-zero.q

echo
echo "grover128-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover128-plus.q

echo
echo "grover150-zero; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover150-zero.q

echo
echo "grover150-plus; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover150-plus.q

