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
Write the hubbard operators in pauli strings
"""

from copy import deepcopy
import numpy as np

def generate_hopping(regs, link_idx, species):
    """
    Generate the hopping operators of the hamiltonian given the
    jordan-wigner transformation. Thus, the output are Pauli strings.

    Parameters
    ----------
    regs : dict
        Dictionary of the SiteRegisters
    link_idx : tuple
        Unique identifier of the link where the hopping will take place
    species : str
        Matter species involved in the hopping

    Example
    -------
    We report here an example of the link numbering

    .. code-block::

          q-h6-q-h7-q-h8-q
          |    |    |    |
         v4    v5   v6   v7
          |    |    |    |
          q-h3-q-h4-q-h5-q
          |    |    |    |
         v0    v1   v2   v3
          |    |    |    |
          q-h0-q-h1-q-h2-q
    """
    # If the y component of the index is even
    # the link is horizontal, otherwise vertical
    is_horizontal = (link_idx[1] == 'h')
    link_idx = int(link_idx[2:])
    shape = regs.shape
    if is_horizontal:
        ypos = link_idx//(shape[0]-1)
        xpos = link_idx%(shape[0]-1)
        from_site_reg = regs[ f'q({xpos}, {ypos})' ]
        to_site_reg = regs[ f'q({xpos+1}, {ypos})' ]
    else:
        ypos = link_idx//shape[0]
        xpos = link_idx%shape[0]
        from_site_reg = regs[ f'q({xpos}, {ypos})' ]
        to_site_reg = regs[ f'q({xpos}, {ypos+1})' ]

    from_site_list = np.array(['I' for _ in from_site_reg.map])
    to_site_list = np.array(['I' for _ in to_site_reg.map])

    from_site_list[ from_site_reg.map[species] ] = 'X'
    to_site_list[ to_site_reg.map[species] ] = 'Y'

    # Check which type of hopping it is
    # Hopping on x axis
    if is_horizontal:
        # Hopping to the right
        if from_site_reg.pos[0] < to_site_reg.pos[0]:
            from_rishon = 'e'
            to_rishon = 'w'
        # Hopping to the left
        else:
            from_rishon = 'w'
            to_rishon = 'e'
    # Hopping on y axis
    else:
        # Hopping to the top
        if from_site_reg.pos[1] < to_site_reg.pos[1]:
            from_rishon = 'n'
            to_rishon = 's'
        # Hopping to the bottom
        else:
            from_rishon = 's'
            to_rishon = 'n'

    # In the following we put an X on from/to_rishon even though the correct general
    # term has a Y. This is because a YY interaction is mapped into an X interaction
    # when we reduce the mapping. Thus, for the correct functioning of the simulation
    # we already put the reduced X
    from_site_list[from_site_reg.map[species]+1:from_site_reg.map[from_rishon]] = 'Z'
    from_site_list[from_site_reg.map[from_rishon]] = 'X'

    to_site_list[to_site_reg.map[species]+1:to_site_reg.map[to_rishon]] = 'Z'
    to_site_list[to_site_reg.map[to_rishon]] = 'X'

    # Obtain the tensor product between the two hilber spaces by appending the operators
    operator =  np.hstack( (from_site_list, ['⊗'], to_site_list) )

    # Add the hermitian conjugate
    herm_conj = deepcopy(operator)
    herm_conj[from_site_reg.map[species]] = 'Y'
    herm_conj[len(from_site_list) + to_site_reg.map[species]+1] = 'X'

    operator = { ''.join(operator) : 0.5, ''.join(herm_conj) : -0.5 }

    return operator, (from_site_reg, to_site_reg)

def from_operators_to_pauli_dict(pauli_hamiltonian):
    """
    Transform a Hamiltonian described as a dict, where the keys
    are the pauli strings and the values the coefficients into
    a pauli dict that can be read by the qiskit class
    :py:class:`WeightedPauliOperator`.

    Parameters
    ----------
    pauli_hamiltonian : dict
        Pauli hamiltonian. the keys are the pauli strings and
        the values the coefficients

    Returns
    -------
    dict
        Pauli dict to be used in `WeightedPauliOperator.from_dict`
    """
    paulis = []
    for label, coeff in pauli_hamiltonian.items():
        paulis += [
            {
                "coeff": {"real": np.real(coeff)},
                "label": label
            }
        ]

    return { 'paulis': paulis }