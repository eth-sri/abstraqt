circuit: 3 qubits

X(2)

H(0)
H(1)
H(2)

// U_f 
// empty

H(0)
H(1)

assert state in span { |-00> , |-00> }

measure 0..2
