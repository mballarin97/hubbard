import numpy as np

def lattice_str(state, regs, shape, use_1d_map=False):
    """
    Prints a *dense_state* with kets, Following the numbering
    of the registers

    Parameters
    ----------
    state: array_like or dict
        Dense representation of a quantum state
    regs: dict
        Dictionary of sites register
    shape: tuple
        Shape of the hubbard lattice
    use_1d_map : bool, optional
        If True, use the 1d map when sorting out indexes. Default to False.

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
            lat_state = lattice_state(binaries[ii], regs, shape, use_1d_map)
            if np.isclose(np.imag(coef), 0.):
                if np.isclose(np.real(coef), 1.):
                    lattice.append( ['',  lat_state] )
                else:
                    lattice.append(['{:.3f}'.format(np.real(coef)),
                        lat_state ])
            else:
                lattice.append(['{:.3f}'.format( coef ),
                        lat_state ])

    lattice_string = ''
    for coef, lat in lattice:
        lattice_string += coef+'\n'
        for y_string_sites in lat[::-1]:
            lattice_string += printable_site_str(y_string_sites)

    return lattice_string

def lattice_state(str_state, regs, shape, use_1d_map=False):
    """
    Starting from a string state, i.e. '0000' return
    the state distributed in the lattice

    Parameters
    ----------
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
    sites = [[] for _ in range(shape[1])]

    old_idx = 0
    for xy, reg in regs.items():
        if use_1d_map:
            current_idxs = [str_state[reg[species] ] for species in reg.map ]
            current_idxs = ''.join(current_idxs)
        else:
            current_idxs = str_state[old_idx:old_idx+len(reg.map)]

        idxs = {}
        for key, val in reg.map.items():
            idxs[key] = current_idxs[val]

        for key in ['n', 's', 'e', 'w']:
            if key not in idxs:
                idxs[key] = None

        xy = xy[1:].replace('(', '').replace(')', '').split(',')
        xy = np.array(xy, dtype=int)
        sites[xy[1]].append( site_str(idxs['u'], idxs['d'], idxs['n'], idxs['s'],
                                idxs['w'], idxs['e'])
        )
        old_idx += len(reg.map)

    lattice = []
    for yy in sites:
        site0 = yy[0]
        for yysite in yy[1:]:
            site0 = stack_horizontally(site0, yysite)
        lattice.append(site0)

    return lattice


def site_str(up, down, north=None, south=None, west=None, east=None,
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
        east_len = 5
        west_len = 2
    elif west is None and east is not None:
        east_len = 8
        west_len = 2
        site_repr['c'] = [f'│{up},{down}├───{east}─']
    elif west is not None and east is None:
        east_len = 5
        west_len = 5
        site_repr['c'] = [f'{west}───┤{up},{down}│']
    else:
        east_len = 8
        west_len = 5
        site_repr['c'] = [f'{west}───┤{up},{down}├───{east}─']

    if north is not None:
        site_repr['n'] = [' '*west_len+ '│'+' '*east_len  ]
        site_repr['n'] += [' '*west_len+ f'{north}'+' '*east_len  ]
        site_repr['n'] += [' '*west_len+ '│'+' '*east_len  ]
        site_repr['n'] += [' '*(west_len-2) +'┌─┴─┐'+ ' '*(east_len-2)  ]
    else:
        site_repr['n'] = [' '*(west_len-2) +'┌───┐' +' '*(east_len-2)  ]

    if south is not None:
        site_repr['s'] = [' '*(west_len-2) +'└─┬─┘' +' '*(east_len-2)  ]
        site_repr['s'] += [' '*west_len+ '│'+' '*east_len  ]
        site_repr['s'] += [' '*west_len+ f'{south}'+' '*east_len  ]
    else:
        site_repr['s'] = [' '*(west_len-2) +'└───┘' +' '*(east_len-2)  ]

    return site_repr

def stack_horizontally(site_str0, site_str1):
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

def printable_site_str(site_str):
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