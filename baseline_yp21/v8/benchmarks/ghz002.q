// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 2 qubits

// Init
H(0)
X(1)

// Two CNOTs
CNOT(0,1)

// finish up
H(0)
H(1)

assert state in span { |00> , |11> }

measure 0..1
