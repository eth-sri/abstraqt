// Grover search on 4 qubits and 3 helper qubits

circuit: 7 qubits

// Init
// Nothing!

// Grover iteration 1
// Z_0
// NCNCNOT(0,1,4)
H(4)
CNOT(1,4)
T(4)
CNOT(0,4)
T(4)
CNOT(1,4)
T(4)
CNOT(0,4)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(4)
H(4)

// NCNCNOT(2,3,5)
H(5)
CNOT(3,5)
T(5)
CNOT(2,5)
T(5)
CNOT(3,5)
T(5)
CNOT(2,5)
CNOT(2,3)
D(3)
CNOT(2,3)
D(2)
D(3)
T(5)
H(5)

// CCNOT(4,5,6)
H(6)
CNOT(5,6)
D(6)
CNOT(4,6)
T(6)
CNOT(5,6)
D(6)
CNOT(4,6)
CNOT(4,5)
D(5)
CNOT(4,5)
T(4)
T(5)
T(6)
H(6)

// The key step! 
Z(6)
// CCNOT(4,5,6)
H(6)
CNOT(5,6)
D(6)
CNOT(4,6)
T(6)
CNOT(5,6)
D(6)
CNOT(4,6)
CNOT(4,5)
D(5)
CNOT(4,5)
T(4)
T(5)
T(6)
H(6)

// NCNCNOT(2,3,5)
H(5)
CNOT(3,5)
T(5)
CNOT(2,5)
T(5)
CNOT(3,5)
T(5)
CNOT(2,5)
CNOT(2,3)
D(3)
CNOT(2,3)
D(2)
D(3)
T(5)
H(5)

// NCNCNOT(0,1,4)
H(4)
CNOT(1,4)
T(4)
CNOT(0,4)
T(4)
CNOT(1,4)
T(4)
CNOT(0,4)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(4)
H(4)

H(0)
H(1)
H(2)
H(3)

// Z_0
// NCNCNOT(0,1,4)
H(4)
CNOT(1,4)
T(4)
CNOT(0,4)
T(4)
CNOT(1,4)
T(4)
CNOT(0,4)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(4)
H(4)

// NCNCNOT(2,3,5)
H(5)
CNOT(3,5)
T(5)
CNOT(2,5)
T(5)
CNOT(3,5)
T(5)
CNOT(2,5)
CNOT(2,3)
D(3)
CNOT(2,3)
D(2)
D(3)
T(5)
H(5)

// CCNOT(4,5,6)
H(6)
CNOT(5,6)
D(6)
CNOT(4,6)
T(6)
CNOT(5,6)
D(6)
CNOT(4,6)
CNOT(4,5)
D(5)
CNOT(4,5)
T(4)
T(5)
T(6)
H(6)

// The key step! 
Z(6)
// CCNOT(4,5,6)
H(6)
CNOT(5,6)
D(6)
CNOT(4,6)
T(6)
CNOT(5,6)
D(6)
CNOT(4,6)
CNOT(4,5)
D(5)
CNOT(4,5)
T(4)
T(5)
T(6)
H(6)

// NCNCNOT(2,3,5)
H(5)
CNOT(3,5)
T(5)
CNOT(2,5)
T(5)
CNOT(3,5)
T(5)
CNOT(2,5)
CNOT(2,3)
D(3)
CNOT(2,3)
D(2)
D(3)
T(5)
H(5)

// NCNCNOT(0,1,4)
H(4)
CNOT(1,4)
T(4)
CNOT(0,4)
T(4)
CNOT(1,4)
T(4)
CNOT(0,4)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(4)
H(4)

H(0)
H(1)
H(2)
H(3)

assert state in span { |0000000> , |000++++> }

measure 0..6
