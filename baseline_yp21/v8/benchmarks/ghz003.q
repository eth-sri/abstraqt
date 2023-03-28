// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 3 qubits

// Init
H(0)
H(1)
X(2)

// Two CNOTs
CNOT(1,2)
CNOT(0,2)

// finish up
H(0)
H(1)
H(2)

assert state in span { |000> , |111> }

measure 0..2
