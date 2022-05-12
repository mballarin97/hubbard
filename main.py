from hubbard.circuit import hubbard_circuit
from qiskit import QuantumRegister, ClassicalRegister

qancilla = QuantumRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')

regs, qc = hubbard_circuit((3, 2), qancilla, cancilla)

print(qc)