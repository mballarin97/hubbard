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
import argparse
from scipy.stats import entropy

__all__ = ["hubbard_parser", "lattice_str", "entanglement_entropy", "write_json"]

def hubbard_parser():
    """
    Parser to read from command line the import parameters of the simulation

    Returns
    -------
    argparse.ArgumentParser
        The parser object
    """
    parser = argparse.ArgumentParser(prog='HUBBARD_DIGITAL_EVOLUTION',
        description="""
        Exact simulation of the Hubbard evolution using quantum circuits.
        The Hamiltonian is trotterized and then applied. We are able to measure, at each timestep,
        the kinetic operator, the charges, the entanglement, the plaquette stabilizers and
        the correlation between the occupation of up and down in a site.
        """)
    parser.add_argument('--shape', nargs='+', default=(2,2), type=int,
        help='Lattice shape, given as tuple. Default to (2, 2).')
    parser.add_argument('--dt', nargs='?', const=0.01, type=float, default=0.01,
        help='Duration of a single timestep. Default to 0.01.')
    parser.add_argument('--num_trotter_steps', nargs='?', const=10, type=int, default=10,
        help='Number of trotter steps for the simulation of a single timestep. Default to 10.')
    parser.add_argument('--t', nargs='?', const=1, type=float, default=1,
        help='Hopping constant in the Hamiltonian. Default to 1.')
    parser.add_argument('--Umin', nargs='?', const=-8, default=-8,
        help="""Minimum value of the onsite constant constant in the Hamiltonian.""")
    parser.add_argument('--Umax', nargs='?', const=-1/8, default=-1/8,
        help="""Maximum value of the onsite constant constant in the Hamiltonian.""")
    parser.add_argument('--Ustep', nargs='?', const=True, default=False,
        help="""If provided, the change from Umax to Umin is a quench at 1/10 of the
            total simulation time.""")
    parser.add_argument('--num_timesteps', nargs='?', const=100, default=100,
        help="""Total number of timesteps in the simulation.""")
    parser.add_argument('--clear', nargs='?', const=True, default=False,
        help="""Delete the data folder and return""")

    return parser

def lattice_str(qc, state, regs, shape):
    """
    Return the string representation of the lattice state, defined
    on a lattice. It assumes the last qubit is the ancilla.

    Parameters
    ----------
    qc : QuantumCircuit
        Hubbard quantum circuit to retrieve the indexes
    state: array_like or dict
        Dense representation of a quantum state
    regs: dict
        Dictionary of sites register
    shape: tuple
        Shape of the hubbard lattice

    Returns
    -------
    str
        String representing the lattice state, that can be printed or saved on file
    """
    if isinstance(state, dict):
        binaries = np.array( list(state.keys() ))
        state = np.array( list(state.values() ))
    else:
        NN = int(np.log2(len(state)))
        binaries = [bin(ii)[2:] for ii in range(2**NN)]
        binaries = ['0'*(NN-len(a)) + a for a in binaries] #Pad with 0s

    lattice = []
    for ii, coef in enumerate(state):
        if not np.isclose(np.abs(coef), 0.):
            # Drop the ancilla qubit
            str_state = binaries[ii][1:]
            lat_state = _lattice_state(qc, str_state, regs, shape)
            if np.isclose(np.imag(coef), 0.):
                if np.isclose(np.real(coef), 1.):
                    lattice.append( ['',  lat_state] )
                else:
                    lattice.append(['{:.9f}'.format(np.real(coef)),
                        lat_state ])
            else:
                lattice.append(['{:.9f}'.format( coef ),
                        lat_state ])

    lattice_string = ''
    for coef, lat in lattice:
        lattice_string += coef+'\n'
        for y_string_sites in lat[::-1]:
            lattice_string += _printable_site_str(y_string_sites)

    return lattice_string

def _lattice_state(qc, str_state, regs, shape):
    """
    Starting from a string state, i.e. '0000' return
    the state distributed in the lattice

    Parameters
    ----------
    qc : QuantumCircuit
        Hubbard quantum circuit to retrieve the indexes
    str_state : str
        State in qiskit format
    regs : dict
        dictionary of SiteRegisters
    shape : tuple
        Shape of the Hubbard lattice
    use_1d_map : bool, optional
        If True, use the 1d map when sorting out indexes. Default to False.s

    Returns
    -------
    list
        List of lattice string. Each list correspond to a y value
    """
    str_state = str_state[::-1]
    avail_sites = [f'q({ii}, {jj})' for ii in range(shape[0]) for jj in range(shape[1])]
    sites = [[] for _ in range(shape[1])]

    old_idx = 0
    already_visited = []
    for site_idx in avail_sites:
        current_reg = regs[site_idx]
        rishons = current_reg._keys

        write_map = {}
        for ii, rr in enumerate(rishons):
            qubit = current_reg[rr]
            if qubit in already_visited:
                write_map[rr] = ''
                continue
            already_visited.append(qubit)
            qidx = qc.find_bit(qubit).index
            write_map[rr] = str_state[qidx]

        for key in ['n', 's', 'e', 'w']:
            if key not in write_map:
                write_map[key] = None

        sites[current_reg.pos[1]].append( _site_str(write_map['u'], write_map['d'],
            write_map['n'], write_map['s'], write_map['w'], write_map['e'])
        )

    lattice = []
    for yy in sites:
        site0 = yy[0]
        for yysite in yy[1:]:
            site0 = _stack_horizontally(site0, yysite)
        lattice.append(site0)

    return lattice


def _site_str(up, down, north=None, south=None, west=None, east=None,
    colored=True):
    """
    Write the site in lattice form

    Parameters
    ----------
    up : int
        Up qubit
    down : int
        down qubit
    north : int, optional
        North rishon, by default None
    south : int, optional
        South rishon, by default None
    west : int, optional
        West rishon, by default None
    east : int, optional
        East rishon, by default None

    Returns
    -------
    dict
        dictionary with the site description
    """
    site_repr = {}
    if west is None and east is None:
        east_len = 4
        west_len = 2
    elif west is None and east is not None:
        east_len = 6
        west_len = 2
        site_repr['c'] = [f'│{up},{down}├───{east}']
    elif west is not None and east is None:
        east_len = 4
        west_len = 5
        site_repr['c'] = [f'{west}───┤{up},{down}│']
    else:
        east_len = 6
        west_len = 5
        site_repr['c'] = [f'{west}───┤{up},{down}├───{east}']

    if north is not None:
        site_repr['n'] = [' '*west_len+ '│'+' '*east_len  ]
        if north != '':
            site_repr['n'] += [' '*west_len+ f'{north}'+' '*east_len  ]
            site_repr['n'] += [' '*west_len+ '│'+' '*east_len  ]
        site_repr['n'] += [' '*(west_len-2) +'┌─┴─┐'+ ' '*(east_len-2)  ]
    else:
        site_repr['n'] = [' '*(west_len-2) +'┌───┐' +' '*(east_len-2)  ]

    if south is not None:
        site_repr['s'] = [' '*(west_len-2) +'└─┬─┘' +' '*(east_len-2)  ]
        if south != '':
            site_repr['s'] += [' '*west_len+ '│'+' '*east_len  ]
            site_repr['s'] += [' '*west_len+ f'{south}'+' '*east_len  ]
    else:
        site_repr['s'] = [' '*(west_len-2) +'└───┘' +' '*(east_len-2)  ]

    return site_repr

def _stack_horizontally(site_str0, site_str1):
    """
    Merge horizontally two sites of the lattice,
    the site should be of the format given by
    the function :func:`site_str`

    Parameters
    ----------
    site_str0 : dict
        First site to be stacked
    site_str1 : dict
        Second site to be stacked (on the right)

    Returns
    -------
    dict
        Stacked site
    """
    stacked_site = {}
    for key in site_str0:
        stacked_site[key] = []
        for elem0, elem1 in zip(site_str0[key], site_str1[key]):
            stacked_site[key] += [elem0+elem1]

    return stacked_site

def _printable_site_str(site_str):
    """
    Pass from the dictionary to a string
    to be directly printed

    Parameters
    ----------
    site_str : dict
        Site dictionary

    Returns
    -------
    str
        String representing the site on the lattice
    """
    pr_site = '\n'.join(site_str['n']) +'\n' + site_str['c'][0] + '\n' + '\n'.join(site_str['s'])
    pr_site += '\n'

    return pr_site

def get_reduced_density_matrix(psi, loc_dim, n_sites, sites,
    print_rho=False):
    """
    Parameters
    ----------
    psi : ndarray
        state of the QMB system
    loc_dim : int
        local dimension of each single site of the QMB system
    n_sites : int
        total number of sites in the QMB system
    site : int or array-like of ints
        Indeces to trace away
    print_rho : bool, optional
        If True, it prints the obtained reduced density matrix], by default False
    Returns
    -------
    ndarray
        Reduced density matrix
    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f'density_mat should be an ndarray, not a {type(psi)}')

    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}')

    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}')

    if np.isscalar(sites):
        sites = [sites]

    # RESHAPE psi
    psi_copy=psi.reshape(*[loc_dim for _ in range(int(n_sites))])
    # DEFINE A LIST OF SITES WE WANT TO TRACE OUT
    indices = -(np.array(sites) - n_sites + 1)

    # COMPUTE THE REDUCED DENSITY MATRIX
    rho = np.tensordot(psi_copy, np.conjugate(psi_copy), axes=(indices, indices))

    # TRANSFORM INTO A MATRIX. THE NUMBER OF ELEMENTS IS 2^{n_sites-num_traced_idxs }
    rho_dim = int( loc_dim**( (n_sites-len(indices)) ) )
    rho = rho.reshape(rho_dim, rho_dim)
    # PRINT RHO
    if print_rho:
        print('----------------------------------------------------')
        print(f'DENSITY MATRIX TRACING SITES ({str(indices)})')
        print('----------------------------------------------------')
        print(rho)

    return rho

def von_neumann_entropy(eigvs, base = np.e):
    """
    Compute the Von Neumann entanglement entropy of a density matrix
    with eigenvalues :math:`\\lambda_i`
    .. math::
        S_V = -Tr(\\rho\ln\\rho)=-\\sum_{i} \\lambda_i\\log \\lambda_i
    Parameters
    ----------
    eigvs : array-like of floats
        Eigenvalues of the density matrix
    Returns
    -------
    float
        entanglement entropy
    """

    # Clean from negative negligible eigenvalues (zero up to machine precision)
    eigvs = np.array([max(0., eig) for eig in eigvs])

    entanglement = entropy(eigvs, base = base) #-np.sum( eigvs*np.log(eigvs) )


    return entanglement

def entanglement_entropy(statevector, idx_to_trace=None):
    """
    Entanglement entropy of subsystem of a pure state.
    Given a statevector (i.e. pure state), builds the density matrix,
    and traces out some systems.
    Then eveluates Von Neumann entropy using Qiskit's implementation.
    Be consistent with the base of the logarithm.
    Parameters
    ----------
    statevector : array-like
        Statevector of the system
    idx_to_trace : array-like, optional
        Indexes to trace away. By default None.
    Returns
    -------
    float
        Entanglement entropy of the reduced density matrix obtained from statevector
        by tracing away the indexes selected
    """

    num_sites = int( np.log2(len(statevector)) )

    # Construct density matrix
    partial_rho = get_reduced_density_matrix(statevector, 2, num_sites, idx_to_trace)

    # get eigenvalues of partial_rho
    eigvs, _ = np.linalg.eigh(partial_rho)

    ent_entropy = von_neumann_entropy(eigvs)

    return ent_entropy

def write_json(dictionary, fh):
    """
    Write in a pretty format a dictionary
    to a file in json format

    Parameters
    ----------
    dictionary : dict
        dictionary to write on file
    fh : filehandle
        Filehandle to write on file
    """
    string = ''
    string += '{\n'
    for key, value in dictionary.items():
        if isinstance(value, tuple) or isinstance(value, np.ndarray):
            value = list(value)
        elif isinstance(value, bool) or isinstance(value, str):
            value = f"\"{value}\""
        string += f'\t\"{key}\": {value},\n'
    string = string[:-2] + '\n}'
    fh.write(string)
