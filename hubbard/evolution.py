# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import numpy as np

import pickle
from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Parameter
from qmatchatea.preprocessing import _preprocess_qk, qk_transpilation_params
from .qiskit_pauli import WeightedPauliOperator
from .operators import generate_global_hopping, generate_global_onsite
from .operators import from_operators_to_pauli_dict, generate_chemical_potential
from .circuit import hubbard_circuit

__all__ = ['evolution_operation', 'insert_noise', 'generate_evolution_circuit',
            'adiabatic_operation', 'superposition_adiabatic_operation',
            'add_particles']

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
    vert_links = [f'lv{ii}' for ii in range(shape[0]*(shape[1]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[1]*(shape[0]-1))]
    avail_links = vert_links + horiz_links

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

def generate_evolution_circuit(shape, time_step=0.01, num_trotter_steps=1, hopping_constant=1,
    filename=None):
    """
    Generate and save on file the evolution circuit, already linearized for an MPS evolution.
    The file is `circuit/evol_{shape}.pkl` if it is not passed to the function

    Parameters
    ----------
    shape : tuple
        Shape of the lattice
    time_step : float, optional
        Time step for the evolution. Also a qiskit Parameter is allowed, by default 0.01
    num_trotter_steps : int, optional
        Number of trotter steps in the decomposition, by default 1
    hopping_constant : float, optional
        Hopping constant for the Hamiltonian. Also a qiskit Parameter is allowed, by default 1
    filename : str, optional
        PATH to the file where to save the circuit. If None, `circuit/evol_{shape}.pkl` is used.
        By default None.
    """
    # Plaquettes definition
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]

    # ============= Initialize qiskit variables =============
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes))]
    regs, qc = hubbard_circuit(shape, qancilla, cancillas )
    evolution_instruction = evolution_operation(qc, regs, shape,
            hopping_constant, Parameter('U'), time_step, num_trotter_steps)
    qc.append(evolution_instruction, range(qc.num_qubits))
    qc = _preprocess_qk(qc, True, optimization=3)
    print(f'====== num_qubits = {qc.num_qubits}')
    print(f'====== Two-qubits gates = {qc.num_nonlocal_gates()}')

    if filename is None:
        if not os.path.isdir('circuits'):
            os.mkdir('circuits')
        filename = f'circuits/evol_{shape}.pkl'
    else:
        qc.qasm(True, filename)

    with open(filename,"wb") as fh:
        pickle.dump(qc, fh)

def adiabatic_operation(qc, regs, shape,
    interaction_constant, onsite_constant,
    chemical_potential,
    dt, num_trotter_steps):
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
    alpha = Parameter('α')
    # Links available in lattice of given shape
    vert_links = [f'lv{ii}' for ii in range(shape[1]*(shape[0]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[0]*(shape[1]-1))]
    avail_links = vert_links + horiz_links

    total_hamiltonian = {}
    # Generate hopping term of Hubbard hamiltonian
    for link_idx in avail_links:
        # Generate the hopping for both the matter species
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant*alpha)
            total_hamiltonian.update(hop_term)
    # Generate on-site term of hubbard hamiltonian
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    for site in sites:
        # ZZ term
        onsite_term = generate_global_onsite(qc, regs, site, onsite_constant*alpha)
        total_hamiltonian.update(onsite_term)
        # Z+Z term
        #onsite_term = generate_starting_onsite(qc, regs, site, -onsite_constant*alpha )
        #total_hamiltonian.update(onsite_term)

    # Generate starting hamiltonian
    for site in sites:
        if (site[0]+site[1]) %2 == 0:
            sign = -1
        else:
            sign = 1
        for mm in ("u", "d"):
            onsite_term = generate_chemical_potential(qc, regs, site, mm, sign*chemical_potential*(1-alpha) )
            total_hamiltonian.update(onsite_term)

    # From dictionary to qiskit pauli_dict
    pauli_dict = from_operators_to_pauli_dict(total_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)
    print(hamiltonian.print_details())

    # Create evolution instruction
    adiabatic_instruction = hamiltonian.evolve_instruction(evo_time=dt,
        expansion_order=2, num_time_slices=num_trotter_steps)#,expansion_mode='suzuki')

    return adiabatic_instruction

def superposition_adiabatic_operation(qc, regs, shape,
    interaction_constant, onsite_constant,
    dt, num_trotter_steps):
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
    alpha = Parameter('α')
    # Links available in lattice of given shape
    vert_links = [f'lv{ii}' for ii in range(shape[0]*(shape[1]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[1]*(shape[0]-1))]
    avail_links = vert_links + horiz_links

    total_hamiltonian = {}
    # Generate hopping term of Hubbard hamiltonian
    for link_idx in avail_links:
        # Generate the hopping for both the matter species
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant*alpha)
            total_hamiltonian.update(hop_term)
    # Generate on-site term of hubbard hamiltonian
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    for site in sites:
        # ZZ term
        onsite_term = generate_global_onsite(qc, regs, site, onsite_constant)
        total_hamiltonian.update(onsite_term)

    # From dictionary to qiskit pauli_dict
    pauli_dict = from_operators_to_pauli_dict(total_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)
    print(hamiltonian.print_details())

    # Create evolution instruction
    adiabatic_instruction = hamiltonian.evolve_instruction(evo_time=dt,
        expansion_order=2, num_time_slices=num_trotter_steps)#,expansion_mode='suzuki')

    return adiabatic_instruction

def add_particles(qc, regs, site0, site1):
    """
    Add a couple of up and down particles in the
    sites site0 and site1. The procedure is different
    is both sites are the same. It is assumed that
    site0 is even and you add an up in site0 and a down
    in site1

    Parameters
    ----------
    qc : QuantumCircuit
        Original quantum circuit
    regs: HubbardRegister
        Collections of quantum registers
    site0: str
        Identifier of the site where to add the up specie.
        It must be even
    site1: str
        Identifier of the site where to add the down specie.
        It must be odd

    Return
    ------
    QuantumCircuit
        Quantum circuit with the gates to create a particle
    """
    if not regs[site0].is_even:
        raise ValueError("site0 must be even")
    temp_qc = QuantumCircuit(*regs.qregisters)
    xx = np.array([ [0, 1], [1, 0]])
    yy = np.array([ [0, -1j], [1j, 0]])
    zz = np.array([ [1, 0], [0, -1]])
    if site0 == site1:
        # Add particles in site (2, 0)
        real_q_idx = [ qc.find_bit(regs[site0]["u"]).index,
                        qc.find_bit(regs[site0]["d"]).index ]

    else:
        # Ordering in the QC is:
        # 1u,1d,1w,1s,1e,1n,2d
        # Real index of the qubits on the quantum circuit
        real_q_idx = []
        for kk in regs[site0].map:
            real_q_idx += [ qc.find_bit(regs[site0][kk]).index ]
        real_q_idx += [ qc.find_bit(regs[site1]["d"]).index ]

    num_qubits = len(real_q_idx)
    z_on_rishons = 1
    for _ in range(num_qubits-2):
        z_on_rishons = np.kron(z_on_rishons, zz)

    xz_on_rishons = np.kron(z_on_rishons, xx)
    yz_on_rishons = np.kron(z_on_rishons, yy)
    first_term = np.kron(zz@xx, xz_on_rishons)
    second_term = -1j*np.kron(zz@yy, xz_on_rishons)
    third_term = -1j*np.kron(zz@xx, yz_on_rishons)
    fourth_term = np.kron(zz@yy, yz_on_rishons)
    matrix = (first_term+second_term+third_term+fourth_term)/2
    temp_qc.append( UnitaryGate(matrix), real_q_idx )
    transp_params = qk_transpilation_params(basis_gates=["u", "rx", "ry", "rz", "cz", "swap"])
    temp_qc = _preprocess_qk(temp_qc, qk_params=transp_params)
    
    return temp_qc
