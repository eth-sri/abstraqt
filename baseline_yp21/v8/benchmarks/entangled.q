circuit: 3 qubits

// Init
H(0)
H(2)
CNOT(0,1)
CNOT(0,1)

assert state in span { |000> , |000> }

measure 0..2
