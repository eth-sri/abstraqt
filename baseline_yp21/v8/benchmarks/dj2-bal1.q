// Deutsch-Jozsa for one of 
// the balanced f : {0,1}^2 -> {0,1}^2

circuit: 3 qubits

X(2)

H(0)
H(1)
H(2)

// U_f
CNOT(0,2)

H(0)
H(1)

measure 0..1
