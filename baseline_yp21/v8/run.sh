#!/bin/bash


echo
echo "bv049; n=50; k=2"
java Main static 2 timed < benchmarks/bv049.q

# echo
# echo "bv099; n=100; k=2"
# java Main static 2 timed < benchmarks/bv099.q

# echo
# echo "bv149; n=150; k=2"
# java Main static 2 timed < benchmarks/bv149.q

# echo
# echo "bv199; n=200; k=2"
# java Main static 2 timed < benchmarks/bv199.q

# echo
# echo "bv249; n=250; k=2"
# java Main static 2 timed < benchmarks/bv249.q

# echo
# echo "bv299; n=300; k=2"
# java Main static 2 timed < benchmarks/bv299.q


echo
echo "ghz050; n=50; k=2"
java Main static 2 timed < benchmarks/ghz050.q 

# echo
# echo "ghz100; n=100; k=2"
# java Main static 2 timed < benchmarks/ghz100.q

# echo
# echo "ghz150; n=150; k=2"
# java Main static 2 timed < benchmarks/ghz150.q

# echo
# echo "ghz200; n=200; k=2"
# java Main static 2 timed < benchmarks/ghz200.q

# echo
# echo "ghz250; n=250; k=2"
# java Main static 2 timed < benchmarks/ghz250.q

# echo
# echo "ghz300; n=300; k=2"
# java Main static 2 timed < benchmarks/ghz300.q


echo
echo "grover8-zero; n=15; k=5"
java Main static 5 timed ex 3 < benchmarks/single-it-grover8-zero.q

# echo
# echo "grover8-plus; n=15; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover8-plus.q

# echo
# echo "grover32-zero; n=63; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover32-zero.q

# echo
# echo "grover32-plus; n=63; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover32-plus.q

# echo
# echo "grover64-zero; n=127; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover64-zero.q

# echo
# echo "grover64-plus; n=127; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover64-plus.q

# echo
# echo "grover128-zero; n=255; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover128-zero.q

# echo
# echo "grover128-plus; n=255; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover128-plus.q

# echo
# echo "grover150-zero; n=300; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover150-zero-linear.q

# echo
# echo "grover150-plus; n=300; k=5"
# java Main static 5 timed ex 3 < benchmarks/single-it-grover150-plus-linear.q

