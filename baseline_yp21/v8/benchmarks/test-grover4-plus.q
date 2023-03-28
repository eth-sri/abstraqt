// Grover search on 4 qubits and 4 helper qubits

circuit: 8 qubits

// Init
H(0)
H(1)
H(2)
H(3)

// Grover iteration 1
NCNCNOT(0,1,4)
NCCNOT(2,4,5)
NCCNOT(3,5,6)
CNOT(6,7)
// The key step!
Z(7)
CNOT(6,7)
NCCNOT(3,5,6)
NCCNOT(2,4,5)
NCNCNOT(0,1,4)

H(0)
H(1)
H(2)
H(3)

// Grover iteration 1
NCNCNOT(0,1,4)
NCCNOT(2,4,5)
NCCNOT(3,5,6)
CNOT(6,7)
// The key step!
Z(7)
CNOT(6,7)
NCCNOT(3,5,6)
NCCNOT(2,4,5)
NCNCNOT(0,1,4)

assert state in span {
  |00000000> ,
  |0000++++>
}

measure 0..7
