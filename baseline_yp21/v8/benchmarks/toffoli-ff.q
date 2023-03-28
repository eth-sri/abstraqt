// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 3 qubits

// Init
X(0)
X(1)

// Toffoli-ff
H(2)
CNOT(1,2)
T(2)
CNOT(0,2)
T(2)
CNOT(1,2)
T(2)
CNOT(0,2)
CNOT(0,1)
D(1)
CNOT(0,1)
D(0)
D(1)
T(2)
H(2)

// finish up

assert state in span { |011> , |011> }

measure 2..2
