// Grover search on 2 qubits and 1 helper qubits

circuit: 3 qubits

// Init
H(0)
H(1)

// Grover iteration 1
// Z_0
NCNCNOT(0,1,2)
Z(2)
NCNCNOT(0,1,2)

H(0)
H(1)

// Z_0
NCNCNOT(0,1,2)
Z(2)
NCNCNOT(0,1,2)

H(0)
H(1)

assert state in span { |000> , |0++> }

measure 0..2
