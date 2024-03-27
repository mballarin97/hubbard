
from .operators import generate_global_hopping, generate_global_onsite, generate_chemical_potential
from qiskit.circuit import Parameter

def hopping_hamiltonian(qc, regs, shape,
    interaction_constant, parameter=None):
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
    parameter : qiskit.Parameter, optional
        If present, the qiskit parameter is added
        in the definition of the hamiltonian

    Returns
    -------
    dict
        dictionary of pauli strings with their weight
    """
    # Links available in lattice of given shape
    vert_links = [f'lv{ii}' for ii in range(shape[0]*(shape[1]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[1]*(shape[0]-1))]
    avail_links = vert_links + horiz_links

    hamiltonian = {}
    # Generate hopping term of Hubbard hamiltonian
    if isinstance(parameter, Parameter):
        interaction_constant = interaction_constant*parameter

    for link_idx in avail_links:
        # Generate the hopping for both the matter species
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hamiltonian.update(hop_term)

    return hamiltonian


def onsite_hamiltonian(qc, regs, shape,
    onsite_constant, parameter=None):
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
    parameter : qiskit.Parameter, optional
        If present, the qiskit parameter is added
        in the definition of the hamiltonian

    Returns
    -------
    dict
        dictionary of pauli strings with their weight
    """
    # Generate on-site term
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]

    hamiltonian = {}
    # Generate hopping term of Hubbard hamiltonian
    if isinstance(parameter, Parameter):
        onsite_constant = onsite_constant*parameter

    for site in sites:
        onsite_term = generate_global_onsite(qc, regs, site, onsite_constant)
        hamiltonian.update(onsite_term)

    return hamiltonian

def chemical_potentials_hamiltonian(qc, regs, shape,
    chemical_potentials, parameter=None):
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
    chemical_potentials : np.ndarray of shape (num_sites, 2)
        Value of the chemical potentials. Each line is the chemical
        potential of a site. The first row is for the up specie, the
        second for the down specie.
    parameter : qiskit.Parameter, optional
        If present, the qiskit parameter is added
        in the definition of the hamiltonian

    Returns
    -------
    dict
        dictionary of pauli strings with their weight
    """
    # Generate on-site term
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    matter = ("u", "d")

    hamiltonian = {}
    # Generate hopping term of Hubbard hamiltonian
    if isinstance(parameter, Parameter):
        onsite_constant = onsite_constant*parameter

    for chemical_potential, site in zip(chemical_potentials, sites):
        for mm, chem_pot_value in zip(matter, chemical_potential):
            if isinstance(parameter, Parameter):
                chem_pot_value = chem_pot_value*parameter
            chemical_potential_term = generate_chemical_potential(qc, regs, site, mm, chem_pot_value)
            hamiltonian.update(chemical_potential_term)

    return hamiltonian
