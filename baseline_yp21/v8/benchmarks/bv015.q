// Bernstein-Vazirani for a = 111...1 (15 bits).

circuit: 16 qubits

// init
X(15)
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

// U_f
CNOT( 0,15)
CNOT( 1,15)
CNOT( 2,15)
CNOT( 3,15)
CNOT( 4,15)
CNOT( 5,15)
CNOT( 6,15)
CNOT( 7,15)
CNOT( 8,15)
CNOT( 9,15)
CNOT(10,15)
CNOT(11,15)
CNOT(12,15)
CNOT(13,15)
CNOT(14,15)

// wrap up
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

assert state in span { |1111111111111111> , |1111111111111111> }

measure 0..14
