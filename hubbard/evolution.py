# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit
from qiskit.aqua.operators import WeightedPauliOperator
import numpy as np
from .operators import generate_hopping, from_operators_to_pauli_dict

__all__ = ['evolution_operation', 'insert_noise']

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

    Return
    ------
    dict
        Dictionary where the keys are the global hopping operator
        and its hermitian conjugate, the values are their coefficients

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

    # List to keep track of already visited qubits
    already_visited = []
    # Generate the global operators, padded with identities
    global_operator = [ ['I']*num_qubs, ['I']*num_qubs ]
    operators = [op.split('⊗') for op in operators]

    # Prepare the global operator of the first dressed site
    rishons = from_site_reg._keys
    for ii, rr in enumerate(rishons):
        qubit = from_site_reg[rr]
        # Keep track of visited qubits to avoid double
        # operations on the shared rishons
        already_visited.append(qubit.register.name)
        # Retrieve the correct index of the qubit
        qidx = qc.find_bit(qubit).index
        for jj in range(2):
            global_operator[jj][qidx] = operators[jj][0][ii]

    # Prepare the global operator of the second dressed site
    rishons = to_site_reg._keys
    for ii, rr in enumerate(rishons):
        qubit = to_site_reg[rr]
        # Skip the already visited rishons
        if qubit.register.name in already_visited:
            continue
        # Retrieve the correct index of the qubit
        qidx = qc.find_bit(qubit).index
        for jj in range(2):
            global_operator[jj][qidx] = operators[jj][1][ii]

    # Pass from list of characters to string, inverting the ordering
    # to satisfy qiskit convention
    global_operator = [''.join(gl[::-1]) for gl in global_operator]

    # Multiply the constants by the value of the coupling
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

    Return
    ------
    dict
        Dictionary where the keys is the global onsite operator,
        the values is its coefficient

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
    # Retrieve the interested register
    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]})' ]
    num_qubs = qc.num_qubits

    # Generate the global operators, padded with identities
    global_operator = ['I']*num_qubs
    # The local operator always has ZZ on the matter and
    # identities elsewhere
    operators =  'ZZ'
    matter_sites = ['u', 'd']
    for ii, matter in enumerate(matter_sites):
        qubit = site_reg[matter]
        # Retrieve qubit index
        qidx = qc.find_bit(qubit).index
        global_operator[qidx] = operators[ii]

    # Pass from list of characters to string, inverting the ordering
    # to satisfy qiskit convention
    global_operator = ''.join(global_operator[::-1])
    return {global_operator: 0.5*potential}

def evolution_operation(qc, regs, shape,
    interaction_constant, onsite_constant, dt, num_trotter_steps):
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
        Time of the evolution operation
    num_trotter_steps : int
        Number of trotter steps for the evolution

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
        # Generate the hopping for both the matter species
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hubbard_hamiltonian.update(hop_term)

    # Generate on-site term
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    for site in sites:
        onsite_term = generate_global_onsite(qc, regs, site, onsite_constant)
        hubbard_hamiltonian.update(onsite_term)

    # From dictionary to qiskit pauli_dict
    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Create evolution instruction
    evolution_instruction = hamiltonian.evolve_instruction(evo_time=dt,
        expansion_order=2, num_time_slices=num_trotter_steps)

    return evolution_instruction

def insert_noise(qc, gates, probs, noisy_gates = None):
    """
    After each gate inside noisy_gates apply an error gate,
    like X, Z or Y with a given probability

    Parameters
    ----------
    qc : QuantumCircuit
        circuit where to inject noise
    gates : array-like of gates
        Noise gates
    probs : array-like of float
        Probability of applying the gate
    noisy_gates : array-like of str, optional
        Gates after which there is a probability of
        applying the noise
    """
    if noisy_gates is None:
        return qc

    gates = np.array(gates, dtype=object)
    probs = np.array(probs)
    noisy_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instruction in qc:
        noisy_qc.append(*instruction)
        if instruction[0].name in noisy_gates:
            rand_u = np.random.uniform(0, 1, len(probs))
            gates_to_apply = gates[rand_u<probs]
            for gate in gates_to_apply:
                site = np.random.choice(instruction[1])
                noisy_qc.append(gate, [site] )

    return noisy_qc
