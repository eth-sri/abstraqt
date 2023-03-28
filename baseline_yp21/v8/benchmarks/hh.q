// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 3 qubits

// Init
H(0)
H(0)
CNOT(1,2)

assert state in span { |000> , |000> }

measure 0..2
