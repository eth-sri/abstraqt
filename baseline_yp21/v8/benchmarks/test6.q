// Grover search on 4 qubits and 3 helper qubits

circuit: 7 qubits

// Init
H(0)
H(1)
H(3)

NCNCNOT(0,1,4)
NCNCNOT(2,3,5)
CCNOT(4,5,6)
Z(6)
CCNOT(4,5,6)
NCNCNOT(2,3,5)
NCNCNOT(0,1,4)

H(0)
H(1)
H(3)

NCNCNOT(0,1,4)
NCNCNOT(2,3,5)
CCNOT(4,5,6)
Z(6)
CCNOT(4,5,6)
NCNCNOT(2,3,5)
NCNCNOT(0,1,4)

H(0)
H(1)

NCNCNOT(0,1,4)
NCNCNOT(2,3,5)
CCNOT(4,5,6)
Z(6)
CCNOT(4,5,6)
NCNCNOT(2,3,5)

assert state in span { |0000000> , |000++++> }

measure 0..6
