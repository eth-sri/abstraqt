#!/bin/bash

echo
echo "bv049; k=2"
java Main static 2 timed < benchmarks/bv049.q

echo
echo "bv099; k=2"
java Main static 2 timed < benchmarks/bv099.q

echo
echo "bv149; k=2"
java Main static 2 timed < benchmarks/bv149.q

echo
echo "bv199; k=2"
java Main static 2 timed < benchmarks/bv199.q

echo
echo "bv249; k=2"
java Main static 2 timed < benchmarks/bv249.q

echo
echo "bv299; k=2"
java Main static 2 timed < benchmarks/bv299.q

