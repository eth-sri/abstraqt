// Program for testing static analysis of NCNCNOT

circuit: 6 qubits

// NCNCNOT(0,1,2)

// X(1)
// CNOT(1,2)
// X(1)

//  /*
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
// */

assert state in span { |000100> , |000100> }

measure 0..5
