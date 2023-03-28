// Grover search on 32 qubits and 31 helper qubits

circuit: 63 qubits

// Init
// Nothing!

// Grover iteration 1
// Z_0
NCNCNOT(0,1,32)
NCNCNOT(2,3,33)
NCNCNOT(4,5,34)
NCNCNOT(6,7,35)
NCNCNOT(8,9,36)
NCNCNOT(10,11,37)
NCNCNOT(12,13,38)
NCNCNOT(14,15,39)
NCNCNOT(16,17,40)
NCNCNOT(18,19,41)
NCNCNOT(20,21,42)
NCNCNOT(22,23,43)
NCNCNOT(24,25,44)
NCNCNOT(26,27,45)
NCNCNOT(28,29,46)
NCNCNOT(30,31,47)
CCNOT(32,33,48)
CCNOT(34,35,49)
CCNOT(36,37,50)
CCNOT(38,39,51)
CCNOT(40,41,52)
CCNOT(42,43,53)
CCNOT(44,45,54)
CCNOT(46,47,55)
CCNOT(48,49,56)
CCNOT(50,51,57)
CCNOT(52,53,58)
CCNOT(54,55,59)
CCNOT(56,57,60)
CCNOT(58,59,61)
CCNOT(60,61,62)
// The key step!
Z(62)
CCNOT(60,61,62)
CCNOT(58,59,61)
CCNOT(56,57,60)
CCNOT(54,55,59)
CCNOT(52,53,58)
CCNOT(50,51,57)
CCNOT(48,49,56)
CCNOT(46,47,55)
CCNOT(44,45,54)
CCNOT(42,43,53)
CCNOT(40,41,52)
CCNOT(38,39,51)
CCNOT(36,37,50)
CCNOT(34,35,49)
CCNOT(32,33,48)
NCNCNOT(30,31,47)
NCNCNOT(28,29,46)
NCNCNOT(26,27,45)
NCNCNOT(24,25,44)
NCNCNOT(22,23,43)
NCNCNOT(20,21,42)
NCNCNOT(18,19,41)
NCNCNOT(16,17,40)
NCNCNOT(14,15,39)
NCNCNOT(12,13,38)
NCNCNOT(10,11,37)
NCNCNOT(8,9,36)
NCNCNOT(6,7,35)
NCNCNOT(4,5,34)
NCNCNOT(2,3,33)
NCNCNOT(0,1,32)

H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
H(14)
H(15)
H(16)
H(17)
H(18)
H(19)
H(20)
H(21)
H(22)
H(23)
H(24)
H(25)
H(26)
H(27)
H(28)
H(29)
H(30)
H(31)

// Z_0
NCNCNOT(0,1,32)
NCNCNOT(2,3,33)
NCNCNOT(4,5,34)
NCNCNOT(6,7,35)
NCNCNOT(8,9,36)
NCNCNOT(10,11,37)
NCNCNOT(12,13,38)
NCNCNOT(14,15,39)
NCNCNOT(16,17,40)
NCNCNOT(18,19,41)
NCNCNOT(20,21,42)
NCNCNOT(22,23,43)
NCNCNOT(24,25,44)
NCNCNOT(26,27,45)
NCNCNOT(28,29,46)
NCNCNOT(30,31,47)
CCNOT(32,33,48)
CCNOT(34,35,49)
CCNOT(36,37,50)
CCNOT(38,39,51)
CCNOT(40,41,52)
CCNOT(42,43,53)
CCNOT(44,45,54)
CCNOT(46,47,55)
CCNOT(48,49,56)
CCNOT(50,51,57)
CCNOT(52,53,58)
CCNOT(54,55,59)
CCNOT(56,57,60)
CCNOT(58,59,61)
CCNOT(60,61,62)
// The key step!
Z(62)
CCNOT(60,61,62)
CCNOT(58,59,61)
CCNOT(56,57,60)
CCNOT(54,55,59)
CCNOT(52,53,58)
CCNOT(50,51,57)
CCNOT(48,49,56)
CCNOT(46,47,55)
CCNOT(44,45,54)
CCNOT(42,43,53)
CCNOT(40,41,52)
CCNOT(38,39,51)
CCNOT(36,37,50)
CCNOT(34,35,49)
CCNOT(32,33,48)
NCNCNOT(30,31,47)
NCNCNOT(28,29,46)
NCNCNOT(26,27,45)
NCNCNOT(24,25,44)
NCNCNOT(22,23,43)
NCNCNOT(20,21,42)
NCNCNOT(18,19,41)
NCNCNOT(16,17,40)
NCNCNOT(14,15,39)
NCNCNOT(12,13,38)
NCNCNOT(10,11,37)
NCNCNOT(8,9,36)
NCNCNOT(6,7,35)
NCNCNOT(4,5,34)
NCNCNOT(2,3,33)
NCNCNOT(0,1,32)

H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
H(14)
H(15)
H(16)
H(17)
H(18)
H(19)
H(20)
H(21)
H(22)
H(23)
H(24)
H(25)
H(26)
H(27)
H(28)
H(29)
H(30)
H(31)

assert state in span {
  |000000000000000000000000000000000000000000000000000000000000000> ,
  |0000000000000000000000000000000++++++++++++++++++++++++++++++++>
}

measure 0..62
