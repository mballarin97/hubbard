from hubbard.new_encoding.registers import HubbardRegister
from hubbard.new_encoding.circuit import hubbard_circuit, initialize_chessboard
from hubbard.new_encoding.stabilizers import apply_plaquette_stabilizers
from hubbard.new_encoding.evolution import generate_hopping, generate_global_hopping
from qiskit import transpile
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator
import numpy as np
from qmatchatea import print_state

from hubbard.qiskit.evolution import generate_global_onsite

# Simulation backend
backend = StatevectorSimulator(precision='double')

# Initialize ancilla bits
qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')
cancilla1 = ClassicalRegister(1, 'ca1')

# Initialize Hubbard state
shape = (2, 2)
regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
qc = initialize_chessboard(qc, regs)
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
qc.barrier()
print(qc)

print(regs['q(1, 0)']['w'] )

hop = generate_hopping(regs, (0, 0), 'u')
print(hop)
global_hop = generate_global_hopping(qc, regs, (0, 0), 'u')
global_onsite = generate_global_onsite(qc, regs, (0, 0))

keys = list(global_hop.keys())
os_keys = list(global_onsite.keys())
for idx, qq in enumerate(qc.qubits):
    print(qq, keys[0][idx], keys[1][idx], os_keys[0][idx])



if False:
    # Simulate the circuit
    res = execute(qc, backend=backend )
    results = res.result()
    counts = results.get_counts()
    #print(counts )
    statevect = results.get_statevector(qc, decimals=3).data

    # Remove the ancilla qubit
    num_sites = int(np.log2(len(statevect)))
    dense_state = statevect.reshape([2]*num_sites)
    dense_state = np.tensordot(dense_state, np.ones(2), ([0], [0]))
    dense_state = dense_state.reshape(2**(num_sites-1) )

    print_state(dense_state)