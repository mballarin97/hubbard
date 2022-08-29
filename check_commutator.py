import hubbard as hbb
from hubbard.evolution import generate_global_hopping, from_operators_to_pauli_dict, WeightedPauliOperator
from qiskit import AncillaRegister, ClassicalRegister
from qiskit.providers.aer import StatevectorSimulator
import numpy as np
from qiskit.aqua.operators import commutator

if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='double')

    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancilla = ClassicalRegister(1, 'ca0')
    cancilla1 = ClassicalRegister(1, 'ca1')

    # Initialize Hubbard state
    shape = (2, 2)
    regs, qc = hbb.hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
    qc = hbb.initialize_chessboard(qc, regs)
    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
    qc.barrier()
    global_hops = []
    # Links available in lattice of given shape
    avail_links = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]+1)]
    avail_links += [(shape[0]-1, jj) for jj in range(shape[1]) if jj%2==1]

    hubbard_hamiltonian = {}
    # Generate hopping term
    for link_idx in avail_links:
        # Generate the hopping for both the matter species
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, 1)
            hubbard_hamiltonian.update(hop_term)

     # From dictionary to qiskit pauli_dict
    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    pauli_dict = from_operators_to_pauli_dict({'IIIIIIIIIXYYX' : 1})
    stabilizer = WeightedPauliOperator.from_dict(pauli_dict)

    print('Hopping term of the hamiltonian:')
    print(hamiltonian.print_details())

    print('Stabilizer term')
    print(stabilizer.print_details())

    print('The hamiltonian and the stabilizers commute:', hamiltonian.commute_with(stabilizer) )
    print('Commutator term')
    print(commutator(hamiltonian, stabilizer).print_details() )
