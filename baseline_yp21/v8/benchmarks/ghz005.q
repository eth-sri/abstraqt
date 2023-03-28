// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 5 qubits

// Init
H(0)
H(1)
H(2)
H(3)
X(4)

// Two CNOTs
CNOT(3,4)
CNOT(2,4)
CNOT(1,4)
CNOT(0,4)

// finish up
H(0)
H(1)
H(2)
H(3)
H(4)

assert state in span { |00000> , |11111> }

measure 0..4
