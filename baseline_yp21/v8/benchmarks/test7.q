// Program for testing static analysis of NCNCNOT

circuit: 6 qubits

NCNCNOT(0,1,2)
NCNCNOT(1,2,3)
H(4)
NCNCNOT(2,3,4)
H(4)

assert state in span { |000100> , |000100> }

measure 0..5
