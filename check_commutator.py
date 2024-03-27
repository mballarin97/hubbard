import hubbard as hbb
from hubbard.evolution import generate_global_hopping, from_operators_to_pauli_dict
#from qiskit.aqua.operators.legacy import WeightedPauliOperator
from qiskit import AncillaRegister, ClassicalRegister
#from qiskit.aqua.operators import commutator
from qiskit_aer.backends import AerSimulator

if __name__ == '__main__':
    # Simulation backend
    backend = AerSimulator(method="statevector", precision='double')

    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancilla = ClassicalRegister(1, 'ca0')
    cancilla1 = ClassicalRegister(1, 'ca1')

    # Initialize Hubbard state
    shape = (4, 2)
    regs, qc = hbb.hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
    print(qc.qregs)
    qc = hbb.initialize_chessboard(qc, regs)
    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
    qc.barrier()
    global_hops = []
    # Links available in lattice of given shape
    vert_links = [f'lv{ii}' for ii in range(shape[0]*(shape[1]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[1]*(shape[0]-1))]
    avail_links = vert_links + horiz_links

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

    print('Hopping term of the hamiltonian:')
    print(hamiltonian.print_details())

    print('')
    print('===================================')
    print('==== PLAQUETTE STABILIZER TERM ====')
    print('===================================')
    plaquettes = [(idx, jdx) for jdx in range(shape[1]-1) for idx in range(shape[0]-1)]
    p_stab = []
    for pidx in plaquettes:
        stabilizer_idx = hbb.plaquette_operator(qc, regs, pidx)
        pauli_dict = from_operators_to_pauli_dict({ stabilizer_idx : 1})
        stabilizer = WeightedPauliOperator.from_dict(pauli_dict)
        p_stab.append(stabilizer)

        # Check commutators on vertexes
        print(f'Stabilizer {stabilizer_idx} term on plaquette {pidx}')
        print('The hamiltonian and the plaquette stabilizers commute:', hamiltonian.commute_with(stabilizer) )
        print('')

    print('')
    print('================================')
    print('==== VERTEX STABILIZER TERM ====')
    print('================================')

    # Generate stabilizers on the vertexes
    for name, site in regs.items():
        stabilizer = list('I'*qc.num_qubits)
        for mm in site._keys:
            qubit = site[mm]
            qidx = qc.find_bit(qubit).index
            stabilizer[qidx] = 'Z'
        stab_id = ''.join(stabilizer[::-1])
        pauli_dict = from_operators_to_pauli_dict({ stab_id : 1})
        stabilizer = WeightedPauliOperator.from_dict(pauli_dict)

        # Check commutators on vertexes
        print(f'Stabilizer {stab_id} term on site {name}')
        print('The hamiltonian and the vertex stabilizers commute:', hamiltonian.commute_with(stabilizer) )
        print('The plaquette stabilizer and the vertex stabilizers commute:',
                all([ps.commute_with(stabilizer) for ps in p_stab]) )
        print('')
