import numpy as np

from qiskit import QuantumRegister, transpile
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator

from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers
from hubbard.qiskit.evolution import generate_global_hopping, from_operators_to_pauli_dict
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

    # Initialize hopping evolution operator
    avail_links = [(0, 0), (0, 1), (1, 1), (0, 2) ]
    hopping_hamiltonian = {}
    for link_idx in avail_links:
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie)
            hopping_hamiltonian.update(hop_term)
    pauli_dict = from_operators_to_pauli_dict(hopping_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Create evolution circuit
    evol_time = -1/(2*np.pi)
    qq = []
    for qreg in qc.qregs:
        if isinstance(qreg, AncillaRegister):
            continue
        for qubit in qreg:
            qq.append(qubit)
    new_qreg = QuantumRegister(bits=qq)
    evolution_instruction = hamiltonian.evolve_instruction(evo_time=evol_time,
        expansion_order=2)
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