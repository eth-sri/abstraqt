// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 3 qubits

// Init
H(0)
CNOT(0,1)
CNOT(1,2)

assert state in span { |000> , |111> }

measure 0..2
