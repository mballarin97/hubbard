from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.aqua.operators import WeightedPauliOperator
from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers
from qiskit import AncillaRegister, ClassicalRegister
from qiskit.providers.aer import StatevectorSimulator
import numpy as np

backend = StatevectorSimulator(precision='single')

qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')
cancilla1 = ClassicalRegister(1, 'ca1')

shape = (2, 2)
regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
qc = initialize_chessboard(qc, regs)
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )


pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": 2}, "label": "ZZZZZZZZ"}
                   ]
}
hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

qiskit_evolve_circuit = hamiltonian.evolve(#state_in=qc,
    quantum_registers=QuantumRegister(8), evo_time=-1/(2*np.pi), expansion_order=2)

# construct the circuit with Qiskit basis gates
qiskit_evolve_circuit = transpile(
    qiskit_evolve_circuit, basis_gates=[ 'u2', 'u3', 'cx', 'h', 'ry', 'rz', 'rx'])

qc.barrier()

qubits = [ q for q in regs['q(0, 0)'].qregister] + [ q for q in regs['q(1, 0)'].qregister]

qc = qc.compose(qiskit_evolve_circuit, qubits)

# draw the circuit
print(qc)