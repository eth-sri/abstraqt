circuit: 3 qubits

X(2)

H(0)
H(1)
H(2)

// U_f
CNOT(0,2)
X(2)

H(0)
H(1)

assert state in span {
  |001> ,
  |101>
}

measure 0..2
