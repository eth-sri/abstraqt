// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 4 qubits

// Init
H(0)
H(1)
H(2)
X(3)

// Two CNOTs
CNOT(2,3)
CNOT(1,3)
CNOT(0,3)

// finish up
H(0)
H(1)
H(2)
H(3)

assert state in span { |0000> , |1111> }

measure 0..3
