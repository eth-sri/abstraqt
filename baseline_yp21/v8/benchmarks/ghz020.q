// GHZ 
// https://quantum-computing.ibm.com/docs/guide/mult-entang/ghz-states

circuit: 20 qubits

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
H(14)
H(15)
H(16)
H(17)
H(18)
X(19)

// Two CNOTs
CNOT(18,19)
CNOT(17,19)
CNOT(16,19)
CNOT(15,19)
CNOT(14,19)
CNOT(13,19)
CNOT(12,19)
CNOT(11,19)
CNOT(10,19)
CNOT( 9,19)
CNOT( 8,19)
CNOT( 7,19)
CNOT( 6,19)
CNOT( 5,19)
CNOT( 4,19)
CNOT( 3,19)
CNOT( 2,19)
CNOT( 1,19)
CNOT( 0,19)

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
H(15)
H(16)
H(17)
H(18)
H(19)

assert state in span {
  |00000000000000000000> ,
  |11111111111111111111>
}

measure 0..19
