// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 15 qubits

// Init
H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
X(14)

// Two CNOTs
CNOT(13,14)
CNOT(12,14)
CNOT(11,14)
CNOT(10,14)
CNOT( 9,14)
CNOT( 8,14)
CNOT( 7,14)
CNOT( 6,14)
CNOT( 5,14)
CNOT( 4,14)
CNOT( 3,14)
CNOT( 2,14)
CNOT( 1,14)
CNOT( 0,14)

// finish up
H(0)
H(1)
H(2)
H(3)
H(4)
H(5)
H(6)
H(7)
H(8)
H(9)
H(10)
H(11)
H(12)
H(13)
H(14)

assert state in span { 
  |000000000000000> , 
  |111111111111111> 
}

measure 0..14
