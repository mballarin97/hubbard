from qcomps import print_state
from hubbard.circuit import hubbard_circuit, initialize_chessboard
from hubbard.stabilizers import apply_plaquette_stabilizers
from qiskit import AncillaRegister, ClassicalRegister, execute, BasicAer

qancilla = AncillaRegister(1, 'a0')
cancilla = ClassicalRegister(1, 'ca0')

regs, qc = hubbard_circuit((2, 2), qancilla, cancilla)
print(regs.keys())
qc = initialize_chessboard(qc, regs)
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla[0], (0,0) )
qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla[0], (0,0) )
print(qc)

res = execute(qc, backend=BasicAer.get_backend('statevector_simulator') )

print_state(res.result().get_statevector(qc, decimals=3) )