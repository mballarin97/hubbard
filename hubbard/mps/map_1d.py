"""
Module for the mapping from 2d hubbard to 1d MPS
"""

def quasi_1d_chain(shape):
    """
    Generate mapping for a quasi-1d chains, meaning that
    has a shape ``shape = (length, 2)``. The output is a list, where
    each element is the tuple ``( 'q(xx, yy)', species)`` where
    species is ('u', 'd', 'n', 'w', 's', 'e').

    The lattice should look like this:

    .. code-block::

        o--o--o--...--o
        |  |  |  ...  |
        o--o--o--...--o


    Parameters
    ----------
    length : int
        Number of sites on the x direction

    Returns
    -------
    list
        1d mapping list

    """
    length = shape[0]
    map_1d = []
    # Loop over x
    for xx in range(length):
        if xx%2 == 0:
            y_order = (0, 1)
        else:
            y_order = (1, 0)

        # Loop over y
        for yy in y_order:
            site_map = _quasi_1d_site(xx, yy, length)
            map_1d += site_map

    return map_1d

def _quasi_1d_site(xx, yy, length):
    """
    Generate the mapping from a quasi 1-dimensional lattice
    of a single site to a 1d map. We recall the arrangement
    for a general bulk site:

    .. code-block::

            n
            |
        w--u-d--e
            |
            s


    Parameters
    ----------
    xx : int
        X axis position
    yy : int
        Y axis position
    length : int
        Number of sites in the x direction

    Returns
    -------
    list
        1d mapping of the site
    """
    if yy > 1:
        raise ValueError('yy coordinate for quasi-1d system must be '+
            f'at most 1, not {yy}')
    site = f'q({xx}, {yy})'

    if yy == 0:
        if xx%2 == 0:
            site_map = ['w', 'u', 'd', 'e', 'n']
        else:
            site_map = ['n', 'w', 'u', 'd', 'e']
    else:
        if xx%2 == 0:
            site_map = ['s', 'w', 'u', 'd', 'e']
        else:
            site_map = ['w', 'u', 'd', 'e', 's']

    # Treat boundary sites differently
    if xx == 0:
        site_map.remove('w')
    elif xx == length-1:
        site_map.remove('e')

    site_map = [ (site, species) for species in site_map ]

    return site_map