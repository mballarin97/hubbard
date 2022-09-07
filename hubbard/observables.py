# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
#from .qiskit_pauli import WeightedPauliOperator
from qiskit.aqua.operators.legacy import WeightedPauliOperator
from .evolution import generate_global_hopping
from .operators import from_operators_to_pauli_dict
from .utils import entanglement_entropy

__all__ = ['compute_kinetic_expectation', 'compute_up_and_down_expectation',
    'compute_updown_expectation', 'compute_entanglement']

def compute_kinetic_expectation(qc, regs, shape, statevect, interaction_constant):
    """
    Measure the kinetik term of the Hamiltonian, which is equivalent to the
    hopping term

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation
    shape : tuple
        Shape of the lattice
    statevect : StateVector
        Statevector describing the quantum state over which we want to measure
        the observable
    interaction_constant : int
        Value of the interaction constant for the kinetic term

    Returns
    -------
    float
        Expectation value of the kinetic term
    """
    # Links available in lattice of given shape
    vert_links = [f'lv{ii}' for ii in range(shape[1]*(shape[0]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[0]*(shape[1]-1))]
    avail_links = vert_links + horiz_links

    # Generate the kinetic term
    hubbard_hamiltonian = {}
    for link_idx in avail_links:
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hubbard_hamiltonian.update(hop_term)
    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Compute the expectation value
    kinetic_expectation = hamiltonian.evaluate_with_statevector(statevect.data)

    # The function returns average and std, but the std is always 0 and we discard it.
    # We keep only the real part because we are measuring a real hamiltonian
    kinetic_expectation = np.real(kinetic_expectation[0])

    return kinetic_expectation

def compute_up_and_down_expectation(qc, regs, statevect):
    """
    Measure the expectation values of up and down matter in each
    sites, which can later be connected to the charge and spin
    densities.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation
    statevect : StateVector
        Statevector describing the quantum state over which we want to measure
        the observable

    Returns
    -------
    list
        list of the expectation values. First, all the up expectations are
        returned, then all the down expectations. The order of measurement
        is the same you get by calling HubbardRegister.values()
    """
    # Total number of qubits
    num_qubs = qc.num_qubits

    up_hamiltonian_exp = []
    down_hamiltonian_exp = []
    matter = ['u', 'd']
    for site in regs.values():
        for mm in matter:
            # Prepare the operator, it just have Z on the up matter
            # of the site
            global_operator = ['I']*num_qubs
            qubit = site[mm]
            qidx = qc.find_bit(qubit).index
            global_operator[qidx] = 'Z'
            # Invert the ordering for qiskit
            hamiltonian = { ''.join(global_operator[::-1]) : 1 }

            pauli_dict = from_operators_to_pauli_dict(hamiltonian)
            hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

            # The function returns average and std, but the std is always 0 and we discard it.
            # We keep only the real part because we are measuring a real hamiltonian
            exp = np.real(hamiltonian.evaluate_with_statevector(statevect)[0] )
            if mm == 'u':
                up_hamiltonian_exp.append( exp )
            else:
                down_hamiltonian_exp.append( exp )

    return up_hamiltonian_exp + down_hamiltonian_exp

def compute_updown_expectation(qc, regs, statevect):
    """
    Measure the expectation values of the PRODUCT up and down matter
    in each site.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation
    statevect : StateVector
        Statevector describing the quantum state over which we want to measure
        the observable

    Returns
    -------
    list
        list of the expectation values. The order of measurement
        is the same you get by calling HubbardRegister.values()
    """
    # Total number of qubits
    num_qubs = qc.num_qubits

    hamiltonian_exp = []
    matter = ['u', 'd']
    for site in regs.values():
        global_operator = ['I']*num_qubs
        for mm in matter:
            # Prepare the operator, it just have Z on the matter
            # of the site
            qubit = site[mm]
            qidx = qc.find_bit(qubit).index
            global_operator[qidx] = 'Z'
        # Invert the ordering for qiskit
        hamiltonian = { ''.join(global_operator[::-1]) : 1 }

        pauli_dict = from_operators_to_pauli_dict(hamiltonian)
        hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

        # The function returns average and std, but the std is always 0 and we discard it.
        # We keep only the real part because we are measuring a real hamiltonian
        exp = np.real(hamiltonian.evaluate_with_statevector(statevect)[0] )

        hamiltonian_exp.append( exp )

    return hamiltonian_exp

def compute_half_entanglement(qc, regs, shape, statevect, base2=True):
    """
    Compute the entanglement by cutting in half
    VERTICALLY the system

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation
    shape : tuple
        Shape of the lattice
    statevect : StateVector
        Statevector describing the quantum state over which we want to measure
        the observable
    base2 : bool, optional
        If true, compute the base2 entropy. Default to True

    Returns
    -------
    float
        Expectation value of the Von Neumann entanglement entropy
    """

    # Cut in half by splitting diagonally on the links,
    # i.e. one rishon link on the left and one on the right
    idxs = [qc.num_qubits-1] # We already put in the ancilla
    cut_on = shape[1]//2
    for site in regs.values():
        rishons = list( site._keys )
        #
        # O-r-O
        # |   |
        # | ----
        # r | r
        # --- |
        # |   |
        # O-r-O
        #
        # Remove the east rishon on odd sites adjacent to the cut
        if site.pos[1]+1 == cut_on and (site.is_even):
            rishons.remove('n')
        # We are interested in only half of the system
        elif site.pos[1]+1 > cut_on:
            continue

        for mm in rishons:
            qubit = site[mm]
            qidx = qc.find_bit(qubit).index
            idxs.append(qidx)

        # Remove double occurrences
    idxs = np.unique(idxs)

    entropy = entanglement_entropy(statevect.data, idxs)
    if base2:
        entropy *= np.log2(np.e)

    return entropy

def compute_links_matter_entanglement(qc, regs, shape, statevect, base2=True):
    """
    Compute the entanglement between links and matter

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation
    shape : tuple
        Shape of the lattice
    statevect : StateVector
        Statevector describing the quantum state over which we want to measure
        the observable
    base2 : bool, optional
        If true, compute the base2 entropy. Default to True

    Returns
    -------
    float
        Expectation value of the Von Neumann entanglement entropy
    """

    # Cut in half by splitting diagonally on the links,
    # i.e. one rishon link on the left and one on the right
    idxs = [qc.num_qubits-1] # We already put in the ancilla
    for site in regs.values():
        for mm in ('u', 'd'):
            qubit = site[mm]
            qidx = qc.find_bit(qubit).index
            idxs.append(qidx)

    idxs = np.sort(idxs)

    entropy = entanglement_entropy(statevect.data, idxs)
    if base2:
        entropy *= np.log2(np.e)

    return entropy
