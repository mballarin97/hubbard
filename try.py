from hubbard.utils import lattice_str
from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers, apply_link_parity_stabilizer, apply_vertex_parity_stabilizer
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator
from qiskit.result import marginal_counts
import numpy as np
from hubbard.operators import generate_hopping
from hubbard.evolution import hopping_circuit

backend = StatevectorSimulator(precision='single')

qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')
cancilla1 = ClassicalRegister(1, 'ca1')

shape = (4, 4)
regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )

op = generate_hopping(regs, (2,2), 'u')
#circ = hopping_circuit((2, 2))

print(op)

#print(circ)