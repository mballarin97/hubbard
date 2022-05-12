from hubbard.utils import lattice_str
from hubbard.circuit import hubbard_circuit, initialize_chessboard
from hubbard.stabilizers import apply_plaquette_stabilizers
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator
from qiskit.result import marginal_counts

backend = StatevectorSimulator(precision='single')

qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')

shape = (2, 2)
regs, qc = hubbard_circuit(shape, qancilla, cancilla)
qc = initialize_chessboard(qc, regs)
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
#qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla[0], (0,0) )
print(qc)

res = execute(qc, backend=backend )
results = res.result()
counts = marginal_counts(results, indices=[0]).get_counts()
print(counts )
statevect = results.get_statevector(qc, decimals=3).data

res = lattice_str(statevect, regs, shape )

print(res)