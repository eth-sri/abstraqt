// Grover search on 8 qubits and 7 helper qubits

circuit: 15 qubits

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
// Z_0
NCNCNOT(0,1,8)
NCNCNOT(2,3,9)
NCNCNOT(4,5,10)
NCNCNOT(6,7,11)
CCNOT(8,9,12)
CCNOT(10,11,13)
CCNOT(12,13,14)
// The key step! 
Z(14)
CCNOT(12,13,14)
CCNOT(10,11,13)
CCNOT(8,9,12)
NCNCNOT(6,7,11)
NCNCNOT(4,5,10)
NCNCNOT(2,3,9)
NCNCNOT(0,1,8)

H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)

// Z_0
NCNCNOT(0,1,8)
NCNCNOT(2,3,9)
NCNCNOT(4,5,10)
NCNCNOT(6,7,11)
CCNOT(8,9,12)
CCNOT(10,11,13)
CCNOT(12,13,14)
// The key step!
Z(14)
CCNOT(12,13,14)
CCNOT(10,11,13)
CCNOT(8,9,12)
NCNCNOT(6,7,11)
NCNCNOT(4,5,10)
NCNCNOT(2,3,9)
NCNCNOT(0,1,8)

H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)


assert state in span { 
  |000000000000000> , 
  |0000000++++++++> 
}

measure 0..14
