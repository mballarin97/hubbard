# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Perform the exact diagonalization of a 2x2 plaquette in order to
check that the spectrum of the hamiltonian is correct
(This is going to be checked with the Jordan-Wigner mapping results)
"""

from hubbard.evolution import generate_global_onsite, generate_global_hopping, from_operators_to_pauli_dict, WeightedPauliOperator, generate_starting_onsite
import numpy as np
import hubbard as hbb
from qiskit import AncillaRegister, ClassicalRegister
import numpy as np
from tqdm import tqdm

def adiabatic_operation(qc, regs, shape,
    interaction_constant, onsite_constant,
    chemical_potential, alpha):
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
        onsite_term = generate_starting_onsite(qc, regs, site, sign*chemical_potential*(1-alpha) )
        total_hamiltonian.update(onsite_term)

    # From dictionary to qiskit pauli_dict
    new_op = {}
    for term, value in total_hamiltonian.items():
        new_op[ term[1:] ] = value
    pauli_dict = from_operators_to_pauli_dict(new_op)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)
    return hamiltonian



shape = (2,2)
# ============= Initialize qiskit variables =============
qancilla = AncillaRegister(1, 'a0')
cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(1)]

# ============= Initialize Hubbard circuit =============
regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )

alphas = np.linspace(0, 1, 100)

for alpha in tqdm(alphas):
    hh = adiabatic_operation(qc, regs, shape, 0.1, -1, -1, alpha)
    big_hh = hh.to_opflow().to_matrix_op()
    eigvl, eigenvector = np.linalg.eigh(big_hh.to_matrix() )


    # Save results
    np.savetxt(f'data/eigvl{alpha}.txt', eigvl)
    np.savetxt(f'data/eigvc{alpha}.txt', eigenvector[:, :10])

