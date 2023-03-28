// Bernstein-Vazirani for a = 1111111111111.

circuit: 14 qubits

// init
X(13)
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

// U_f
CNOT( 0,13)
CNOT( 1,13)
CNOT( 2,13)
CNOT( 3,13)
CNOT( 4,13)
CNOT( 5,13)
CNOT( 6,13)
CNOT( 7,13)
CNOT( 8,13)
CNOT( 9,13)
CNOT(10,13)
CNOT(11,13)
CNOT(12,13)

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

assert state in span { |11111111111111> , |11111111111111> }

measure 0..12
