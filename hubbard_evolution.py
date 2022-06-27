import numpy as np

from qiskit import transpile
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator

from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers
from hubbard.qiskit.evolution import evolution_operation
from hubbard.utils import lattice_str


if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='single')

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

    # Create evolution circuit
    int_const = 1
    site_const = 0
    dt = 0.1
    num_tsteps = 100
    evolution_instruction = evolution_operation(qc, regs, shape, int_const, site_const, dt, num_tsteps)
    qc.append(evolution_instruction, range(qc.num_qubits-1))
    # construct the circuit with Qiskit basis gates
    qc = transpile(qc, basis_gates=[ 'u2', 'u3', 'cx', 'h', 'ry', 'rz', 'rx'])

    # Print some statistics about the quantum circuit
    print('='*10 + ' Circuit statistic ' + '='*10)
    print(f'Lattice shape: {shape}')
    print(f'Number of physical qubits: {qc.num_qubits}')
    print(f'Depth of the circuit: {qc.depth()}')
    print(f'Total number of operations: {qc.count_ops()}')
    print(f'Total number of non-local gates: {qc.num_nonlocal_gates()}')
    print('='*39)

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


    res = lattice_str(statevect, regs, shape )

    print( res )