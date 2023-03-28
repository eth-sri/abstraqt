// Grover search on 2 qubits and 1 helper qubits

circuit: 3 qubits

// Init
H(0)
H(1)

// Grover iteration 1
// Z_0
// NCNCNOT(0,1,2)
H(2)
CNOT(1,2)
T(2)
CNOT(0,2)
T(2)
CNOT(1,2)
T(2)
CNOT(0,2)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(2)
H(2)

Z(2)
// NCNCNOT(0,1,2)
H(2)
CNOT(1,2)
T(2)
CNOT(0,2)
T(2)
CNOT(1,2)
T(2)
CNOT(0,2)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(2)
H(2)


H(0)
H(1)

// Z_0
// NCNCNOT(0,1,2)
H(2)
CNOT(1,2)
T(2)
CNOT(0,2)
T(2)
CNOT(1,2)
T(2)
CNOT(0,2)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(2)
H(2)

Z(2)
// NCNCNOT(0,1,2)
H(2)
CNOT(1,2)
T(2)
CNOT(0,2)
T(2)
CNOT(1,2)
T(2)
CNOT(0,2)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(2)
H(2)


H(0)
H(1)

assert state in span { |000> , |0++> }

measure 0..2
