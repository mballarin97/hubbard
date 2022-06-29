# code from https://github.com/DavitKhach/quantum-algorithms-tutorials/blob/master/Hamiltonian_simulation.ipynb

from copy import deepcopy
from qiskit.aqua.operators import WeightedPauliOperator
import numpy as np
from ..operators import generate_hopping
from ..qiskit.evolution import from_operators_to_pauli_dict
from qiskit.extensions.quantum_initializer.initializer import initialize
from qiskit.providers.aer import StatevectorSimulator

def generate_global_hopping(qc, regs, link_idx, species, coupling=1):
    """
    Generate the hopping operators of the hamiltonian given the
    jordan-wigner transformation. Thus, the output are Pauli strings.
    The operator is global in the sense that is padded with identities.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        Quantum circuit class containing the Hubbard circuit
    regs : dict
        Dictionary of the SiteRegisters
    link_idx : tuple
        Unique identifier of the link where the hopping will take place
    species : str
        Matter species involved in the hopping
    coupling : float, optional
        Coupling in the link

    Example
    -------
    We report here an example of the link numbering

    .. code-block::

          q-(0,4)-q-(1,4)-q-(2,4)-q
          |       |       |       |
        (0,3)   (1,3)   (2,3)   (3,3)
          |       |       |       |
          q-(0,2)-q-(1,2)-q-(2,2)-q
          |       |       |       |
        (0,1)   (1,1)   (2,1)   (3,1)
          |       |       |       |
          q-(0,0)-q-(1,0)-q-(2,0)-q
    """
    # Generate the local operator, defined only on the interested
    # registers
    local_operators, (from_site_reg, to_site_reg) = generate_hopping(regs, link_idx, species)
    operators = local_operators.keys()
    num_qubs = qc.num_qubits

    # Generate the global operators, padded with identities
    already_visited = []
    global_operator = [ ['I']*num_qubs, ['I']*num_qubs ]
    operators = [op.split('⊗') for op in operators]
    rishons = from_site_reg._keys
    for ii, rr in enumerate(rishons):
        qubit = from_site_reg[rr]
        already_visited.append(qubit.register.name)
        qidx = qc.find_bit(qubit).index
        for jj in range(2):
            global_operator[jj][qidx] = operators[jj][0][ii]

    rishons = to_site_reg._keys
    for ii, rr in enumerate(rishons):
        qubit = to_site_reg[rr]
        if qubit.register.name in already_visited:
            continue
        qidx = qc.find_bit(qubit).index
        for jj in range(2):
            global_operator[jj][qidx] = operators[jj][1][ii]

    global_operator = [''.join(gl) for gl in global_operator]
    values = coupling * np.array(list(local_operators.values()))
    return dict(zip(global_operator, values) )

def generate_global_onsite(qc, regs, site_idx, potential=1):
    """
    Generate the onsite operators of the hamiltonian given the
    jordan-wigner transformation. Thus, the output are Pauli strings.
    The operator is global in the sense that is padded with identities.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        Quantum circuit class containing the Hubbard circuit
    regs : dict
        Dictionary of the SiteRegisters
    link_idx : tuple
        Unique identifier of the link where the hopping will take place
    species : str
        Matter species involved in the hopping
    potential : 1
        Value of the coefficient of the potential

    Example
    -------
    We report here an example of the vertex numbering

    .. code-block::

        (0,2)--(1,2)--(2,2)
          |      |      |
        (0,1)--(1,1)--(2,1)
          |      |      |
        (0,0)--(1,0)--(2,0)
    """
    # Generate the local operator, defined only on the interested
    # registers
    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]})' ]
    num_qubs = qc.num_qubits-1

    # Generate the global operators, padded with identities
    global_operator = ['I']*num_qubs
    operators =  'ZZ'+ 'I'*(len(site_reg.map)-2)
    for ii, qubit in enumerate(site_reg.qregister):
        qidx = qc.find_bit(qubit).index
        global_operator[qidx] = operators[ii]

    global_operator = ''.join(global_operator)
    return {global_operator: 0.5*potential}

def evolution_operation(qc, regs, shape,
    interaction_constant, onsite_constant, dt, num_timesteps):
    """
    Generate the evolution istruction for the Hubbard model

    Parameters
    ----------
    qc : QuantumCircuit
        qiskit hubbard quantum circuit
    regs : dict
        Dictionary of the registers
    shape : tuple
        Shape of the lattice
    interaction_constant : float
        Value of the interaction constant
    onsite_constant : float
        Value of the on-site constant
    dt : float
        Timestep
    num_timesteps : int
        Number of timesteps

    Returns
    -------
    instruction
        qinstruction with the evolution
    """

    # Links available in lattice of given shape
    avail_links = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]+1)]
    avail_links += [(shape[0]-1, jj) for jj in range(shape[1]) if jj%2==1]

    hubbard_hamiltonian = {}
    # Generate hopping term
    for link_idx in avail_links:
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hubbard_hamiltonian.update(hop_term)
    # Generate on-site term
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    for site in sites:
        onsite_term = generate_global_onsite(qc, regs, site, onsite_constant)
        hubbard_hamiltonian.update(onsite_term)

    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Create evolution instruction
    evolution_instruction = hamiltonian.evolve_instruction(evo_time=dt,
        expansion_order=2, num_time_slices=num_timesteps)

    return evolution_instruction

def compute_expectation(qc, regs, shape, statevect, interaction_constant):

    backend = StatevectorSimulator(precision='double')
    # Links available in lattice of given shape
    avail_links = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]+1)]
    avail_links += [(shape[0]-1, jj) for jj in range(shape[1]) if jj%2==1]

    hubbard_hamiltonian = {}
    for link_idx in avail_links:
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hubbard_hamiltonian.update(hop_term)

    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Remove the ancilla qubit
    hamiltonian_expectation = hamiltonian.evaluate_with_statevector(statevect)

    return hamiltonian_expectation
