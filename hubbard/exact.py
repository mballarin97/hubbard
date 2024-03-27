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
from more_itertools import distinct_permutations
from itertools import product
from copy import deepcopy

__all__ = ['all_possible_matter_states', 'new_state_index', 'all_possible_matter_states_opt']

def is_updown_number_respected(state, num_up, num_down):
    """
    Check if a given state respect the symmetry
    of the number of up and down species seperatly

    Parameters
    ----------
    state: np.ndarray of ints
        State of the matter of the system. It is assumed
        a structure like [u, d, u, d, u, d, ...].
    num_up: int
        Number of up specie in the system
    num_down: int
        Number of down specie in the system

    Returns
    -------
    bool
        If True, the state has the correct number
        of up/down specie, otherwise False.
    """
    num_tot = len(state)
    state = np.array(state, dtype=int)
    # Number of up in the state
    ups = state[np.arange(0, num_tot, 2)].sum()
    # Number of down in the state
    downs = state[np.arange(1, num_tot, 2)].sum()

    if num_up != ups or num_down != downs:
        flag = False
    else:
        flag = True

    return flag

def all_possible_matter_states(shape, num_up, num_down):
    """
    Return all possible states of the matter of a lattice
    of a given shape with a given number of up/down species.
    The states returned are a 1-dimensional array where each
    site is defined by two numbers, and we always report first
    the up specie and then the down specie (differently from
    the mapping)

    Parameters
    ----------
    shape : array-like
        Shape (x, y) of the lattice
    num_up : int
        Number of up species
    num_down : int
        Number of down species

    Returns
    -------
    np.ndarray
        Array of all sites that respect the num_up and num_down
        constraint. Each row is a state.
    """
    # Number of matter is number of sites*2
    num_mm_sites = 2*np.prod(shape)
    # Generate a state that respect the correct number of
    # particles. Which state is not important, since we will
    # consider all the permutations
    first_state = np.zeros(num_mm_sites, dtype=int)
    first_state[ np.arange(0, num_mm_sites, 2)[:num_up] ] = 1
    first_state[ np.arange(1, num_mm_sites, 2)[:num_down] ] = 1

    # Compute all permutations taking into account that there may
    # be repetitions or wrong permutations
    all_states = first_state
    for state in distinct_permutations(first_state):
        flag = is_updown_number_respected(state, num_up, num_down)
        if flag:
            all_states = np.vstack((all_states, state))

    all_states = all_states[1:, :]
    ordering = all_states.dot(1 << np.arange(all_states.shape[-1] - 1, -1, -1))
    ordering = np.argsort(ordering)

    return all_states[ordering, :]

def all_possible_matter_states_opt(shape, num_up, num_down):
    """
    Return all possible states of the matter of a lattice
    of a given shape with a given number of up/down species.
    The states returned are a 1-dimensional array where each
    site is defined by two numbers, and we always report first
    the up specie and then the down specie (differently from
    the mapping)

    Parameters
    ----------
    shape : array-like
        Shape (x, y) of the lattice
    num_up : int
        Number of up species
    num_down : int
        Number of down species

    Returns
    -------
    np.ndarray
        Array of all sites that respect the num_up and num_down
        constraint. Each row is a state.
    """
    # Number of matter is number of sites
    num_sites = np.prod(shape)
    # Generate a state that respect the correct number of
    # particles. Which state is not important, since we will
    # consider all the permutations
    up_state = np.zeros(num_sites, dtype=int)
    up_state[:num_up] = 1
    down_state = np.zeros(num_sites, dtype=int)
    down_state[:num_down] = 1

    # Compute all permutations taking into account that there may
    # be repetitions or wrong permutations
    all_up_states = []
    for state in distinct_permutations(up_state):
        all_up_states.append(state)
    all_down_states = []
    if num_down == num_up:
        all_down_states = all_up_states
    else:
        for state in distinct_permutations(down_state):
            all_down_states.append(state)

    all_states = []
    for up_state, down_state in product(all_up_states, all_down_states):
        all_states.append( np.stack((up_state, down_state), axis=1).ravel())
    all_states = np.array(all_states)

    all_states = all_states[1:, :]
    ordering = all_states.dot(1 << np.arange(all_states.shape[-1] - 1, -1, -1))
    ordering = np.argsort(ordering)

    return all_states[ordering, :]

def new_state_index(all_states, shape, input_idx, paulis, site_ordering):
    """
    Given an input state check without any contraction which state will
    be the output. It should increase the performances of the algorithm.

    Parameters
    ----------
    all_states : np.ndarray
        List of all possible states that satisfy the symmetry
    shape : array-like
        Shape of the lattice
    input_idx : int
        Index of the input state
    paulis : str
        Hamiltonian term to apply to new_state_index
    site_ordering : array-like
        Array to reorder the site in the same convention of all_states

    Retuns
    ------
    int
        Index of the new state w.r.t. all_states
    """
    num_links = shape[0]*(shape[1]-1) + shape[1]*(shape[0]-1)
    paulis = np.array(list(paulis))
    paulis = paulis[1:-num_links][::-1]
    paulis = paulis[site_ordering]

    input_state = all_states[input_idx]
    output_state = deepcopy(input_state)
    output_state[np.logical_or(paulis=="X", paulis=="Y") ] -= 1
    output_state = np.abs(output_state)

    for idx, state in enumerate(all_states):
        if np.isclose(state, output_state).all():
            return idx

