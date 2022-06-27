from hubbard.utils import lattice_str
from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers, apply_link_parity_stabilizer, apply_vertex_parity_stabilizer
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, execute 
from qiskit.providers.aer import StatevectorSimulator
from qiskit.result import marginal_counts
import numpy as np
from hubbard.operators import generate_hopping
from hubbard.evolution import hopping_circuit
from hubbard.qiskit.evolution import generate_global_hopping

backend = StatevectorSimulator(precision='single')

qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')
cancilla1 = ClassicalRegister(1, 'ca1')

shape = (2, 2)
regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )

qubits = [ ii for ii in qc.qubits ]
op = generate_hopping(regs, (0,1), 'u')
#circ = hopping_circuit((2, 2))

print(op)

global_op = generate_global_hopping(qc, regs, (0, 1), 'u')
print(global_op)
#print(circ)