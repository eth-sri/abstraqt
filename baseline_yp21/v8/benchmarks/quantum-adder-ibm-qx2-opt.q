// Quantum adder on IBM QX2

circuit: 4 qubits

// Init
X(3)
X(2)
H(1)
CNOT(0,1)
T(3)
T(2)
T(0)
D(1)

// Box
CNOT(3,2)
CNOT(0,1)
CNOT(2,0)
CNOT(3,2)
CNOT(1,2)
CNOT(2,3)
CNOT(0,1)

// Finish up
D(3)
D(2)
D(0)
T(1)
CNOT(3,2)
CNOT(0,1)
S(1)
CNOT(1,3)
H(1)

// assert state in span { |000> , |111> }

measure 0..3
