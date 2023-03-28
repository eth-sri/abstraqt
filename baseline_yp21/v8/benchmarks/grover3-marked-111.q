circuit: 3 qubits

// http://iontrap.umd.edu/wp-content/uploads/2012/12/s41467-017-01904-7.pdf
// Figure 1.e: grover3 with two marked states, |011> and |101>
// CCZ decomposed with 
// https://www.researchgate.net/figure/Circuit-diagram-decomposition-of-the-three-qubit-ccz-gate-in-terms-of-seven-single-qubit_fig1_312023141

// Init
H(0)
H(1)
H(2)

// Oracle
// CCZ(0,1,2)
CNOT(1,2)
D(2)
CNOT(0,2)
T(2)
CNOT(1,2)
D(2)
CNOT(0,2)
T(1)
T(2)
CNOT(0,1)
T(0)
D(1)
CNOT(0,1)

// Amplication
H(0)
H(1)
H(2)
X(0)
X(1)
X(2)

// CCZ(0,1,2)
CNOT(1,2)
D(2)
CNOT(0,2)
T(2)
CNOT(1,2)
D(2)
CNOT(0,2)
T(1)
T(2)
CNOT(0,1)
T(0)
D(1)
CNOT(0,1)

// wrap up
X(0)
X(1)
X(2)
H(0)
H(1)
H(2)

assert state in span { |111> , |+++> }

measure 0..2
