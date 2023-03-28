// Bernstein-Vazirani for a = 110.

circuit: 4 qubits

// init
X(3)
H(0)
H(1)
H(2)
H(3)

// U_f
CNOT(0,3)
CNOT(1,3)

// wrap up
H(0)
H(1)
H(2)

assert state in span { |110-> , |110-> }

measure 0..2
