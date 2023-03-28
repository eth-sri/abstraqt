#!/bin/bash

echo
echo "ghz050; k=2"
java Main static 2 timed < benchmarks/ghz050.q 

echo
echo "ghz100; k=2"
java Main static 2 timed < benchmarks/ghz100.q

echo
echo "ghz150; k=2"
java Main static 2 timed < benchmarks/ghz150.q

echo
echo "ghz200; k=2"
java Main static 2 timed < benchmarks/ghz200.q

echo
echo "ghz250; k=2"
java Main static 2 timed < benchmarks/ghz250.q

echo
echo "ghz300; k=2"
java Main static 2 timed < benchmarks/ghz300.q


