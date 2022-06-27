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
    # If the y component of the index is even
    # the link is horizontal, otherwise vertical
    is_horizontal = (link_idx[1]%2 == 0)
    if is_horizontal:
        from_site_reg = regs[ f'q({link_idx[0]}, {link_idx[1]//2})' ]
        to_site_reg = regs[ f'q({link_idx[0]+1}, {link_idx[1]//2})' ]
    else:
        from_site_reg = regs[ f'q({link_idx[0]}, {link_idx[1]-1})' ]
        to_site_reg = regs[ f'q({link_idx[0]}, {link_idx[1]})' ]

    from_site_list = np.array(['I' for _ in from_site_reg.map])
    to_site_list = np.array(['I' for _ in to_site_reg.map])

    from_site_list[ from_site_reg.map[species] ] = 'Y'
    to_site_list[ to_site_reg.map[species] ] = 'X'

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

    from_site_list[from_site_reg.map[species]+1:from_site_reg.map[from_rishon]] = 'Z'
    from_site_list[from_site_reg.map[from_rishon]] = 'X'

    to_site_list[to_site_reg.map[species]+1:to_site_reg.map[to_rishon]] = 'Z'
    to_site_list[to_site_reg.map[to_rishon]] = 'X'

    # Obtain the tensor product between the two hilber spaces by appending the operators
    operator =  np.hstack( (from_site_list, ['⊗'], to_site_list) )

    # Add the hermitian conjugate
    herm_conj = deepcopy(operator)
    herm_conj[from_site_reg.map[species]] = 'X'
    herm_conj[len(from_site_list) + to_site_reg.map[species]+1] = 'Y'

    operator = { ''.join(operator) : 0.5, ''.join(herm_conj) : -0.5 }

    return operator, (from_site_reg, to_site_reg)

