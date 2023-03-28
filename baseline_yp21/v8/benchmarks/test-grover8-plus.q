// Grover search on 8 qubits and 8 helper qubits

circuit: 16 qubits

// Init
H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)

// Grover iteration 1
X(8)
NCCNOT(0,8,9)
NCCNOT(1,9,10)
NCCNOT(2,10,11)
NCCNOT(3,11,12)
NCCNOT(4,12,13)
NCCNOT(5,13,14)
NCCNOT(6,14,15)
NCCNOT(7,15,16)
// The key step!
Z(15)
NCCNOT(7,14,15)
NCCNOT(6,13,14)
NCCNOT(5,12,13)
NCCNOT(4,11,12)
NCCNOT(3,10,11)
NCCNOT(2,9,10)
NCCNOT(1,8,9)
X(8)
CNOT(0,8)
H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)

CNOT(0,8)
X(8)
NCCNOT(1,8,9)
NCCNOT(2,9,10)
NCCNOT(3,10,11)
NCCNOT(4,11,12)
NCCNOT(5,12,13)
NCCNOT(6,13,14)
NCCNOT(7,14,15)
// The key step!
Z(15)
NCCNOT(7,14,15)
NCCNOT(6,13,14)
NCCNOT(5,12,13)
NCCNOT(4,11,12)
NCCNOT(3,10,11)
NCCNOT(2,9,10)
NCCNOT(1,8,9)
X(8)
CNOT(0,8)
H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)

assert state in span {
  |0000000000000000> ,
  |00000000++++++++>
}

measure 0..15
