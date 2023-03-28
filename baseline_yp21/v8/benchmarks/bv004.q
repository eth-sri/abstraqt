// Bernstein-Vazirani for a = 1111.

circuit: 5 qubits

// init
X(4)
H(0)
H(1)
H(2)
H(3)
H(4)

// U_f
CNOT( 0,4)
CNOT( 1,4)
CNOT( 2,4)
CNOT( 3,4)

// wrap up
H(0)
H(1)
H(2)
H(3)
H(4)

assert state in span { |11111> , |11111> }

measure 0..3
