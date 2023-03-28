// Grover for 
// f : {0,1}^2 -> {0,1}^2
// where f(00) = 1, and f(ab) = 0 otherwise.

circuit: 2 qubits

// Init
H(0)
H(1)

// U_f
X(0)
X(1)
CZ(0,1)
X(0)
X(1)

// middle
H(0)
H(1)

// reflection
Z(0)
Z(1)
CZ(0,1)

// finish up
H(0)
H(1)

assert state in span { |00> , |++> }

measure 0..1
