from hubbard.utils import lattice_str
from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers, apply_link_parity_stabilizer, apply_vertex_parity_stabilizer
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator
from qiskit.result import marginal_counts
import numpy as np

backend = StatevectorSimulator(precision='single')

qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')
cancilla1 = ClassicalRegister(1, 'ca1')

shape = (2, 2)
regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
qc = initialize_chessboard(qc, regs)
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
#qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla[0], (0,0) )


# Apply error on site (0,0) rishon 'n'
qc.x(regs['q(0, 0)']['d'] )
#qc.cx(regs['q(0, 0)']['n'], regs['q(0, 1)']['s'])

# Apply correction through link stabilizer
qc = apply_link_parity_stabilizer(qc, regs, qancilla[0], cancilla1, (0, 1))

# Apply correction through vertex stabilizer
qc = apply_vertex_parity_stabilizer(qc, regs, qancilla[0], cancilla1, (0, 0))

# Apply correction with plaquette
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla1, (0,0) )

#print(qc)
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


res = lattice_str(statevect, regs, shape )

print( res )
#with open(f'lattice_{shape[0]}x{shape[1]}.txt', 'w') as fh:
#    fh.write(res)