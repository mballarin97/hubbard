import uuid

import numpy as np
from qcomps.circuit import Qcircuit, ClassicalCondition

def apply_plaquette_stabilizers(qc, regs, cl_reg, plaquette_idx, correct=True,
    selected_output=None):
    """

    TODO: MOVING ANCILLA FOR MPS

    Apply the stabilizer to a plaquette of the Hubbard defermoinised model,
    recording the result of the projective measurement on a classical
    register

    Parameters
    ----------
    qc : QuantumCircuit
        hubbard quantum  circuit
    regs : dict
        Dictionary of the SiteRegisters
    cl_reg : str
        Name of the classical register where to store the measure
    plaquette_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up
    correct : bool, optional
        If True, apply the correction after the stabilizer measurements.
        Otherwise do nothing. Default to True
    selected_output : int, optional
        A priori selected output for the projective measurement. If None,
        the measurement is draw according to a random number. Default to None.

    Return
    ------
    QuantumCircuit
        The quantum circuit after the application of the stabilizer

    Example
    -------
    We report here an example of the plauqette numbering

    .. code-block::

        q-----q-----q-----q
        | 0,1 | 1,1 | 2,1 |
        q-----q-----q-----q
        | 0,0 | 1,0 | 2,0 |
        q-----q-----q-----q

    """
    if not isinstance(qc, Qcircuit):
        raise TypeError(f'qc must be of type Qcircuit, not {type(qc)} when doing mps '+
            'simulations.')

    if not cl_reg in qc.cregisters:
        qc.add_cregister(cl_reg, 1)

    # Keys are created as:
    # - bottom left
    # - upper left
    # - bottom right
    # - upper right
    # Relative rishons contains the rishons used in the stabilizer
    # based on the corner of the site
    # ENCODING OPTIMIZED FOR QUASI-1D SYSTEMS !!!!
    if plaquette_idx[0] % 2 == 0:
        corner_order = ['bl', 'ul', 'ur', 'br']
        relative_rishons = {
            'bl' : ['e', 'n'],
            'ul' : ['s', 'e'],
            'ur' : ['w', 's'],
            'br' : ['n', 'w'],
        }
    else:
        corner_order = ['ul', 'bl' 'br', 'ur']
        relative_rishons = {
            'ul' : ['e', 's'],
            'bl' : ['n', 'e'],
            'ur' : ['s', 'w'],
            'br' : ['w', 'n'],
        }

    # Site registers that forms the plaquette
    involved_regs = [ f'q({plaquette_idx[0]+ii}, {plaquette_idx[1]+jj})'
        for ii in range(2) for jj in range(2)]
    involved_regs = dict(zip(['bl', 'ul', 'br', 'ur'], involved_regs))

    ancilla = str(uuid.uuid4())
    first_site = regs[involved_regs[corner_order[0]]][relative_rishons[corner_order[0]][0]]
    qc.add_qregister(ancilla, [first_site+1], reference_register='default')
    # Apply hadamard to the ancilla
    qc.h( 0, qreg=ancilla )
    # Apply cx
    for idx, corner in enumerate(corner_order):
        for jdx, rishon in enumerate(relative_rishons[corner]):
            if idx+jdx != 0:
                qc.move_site(0, ancilla, regs[involved_regs[corner]][rishon])
            qc.cx( [0, regs[involved_regs[corner]][rishon]], [ancilla, 'default'] )

    # Measure ancilla on x, i.e. hadamard+measure on z
    qc.h( 0, qreg=ancilla )
    # Apply projective measurement
    qc.measure_projective(0, 0, ancilla, cl_reg, selected_output=selected_output)

    if correct:
        c_if = ClassicalCondition(cl_reg, 1, 0)
        # Apply a controlled x operation
        qc.z(regs[involved_regs['br']]['n'], c_if = c_if)

    # Reset the ancilla
    qc.remove_qregister(ancilla)

    return qc


def apply_vertex_parity_stabilizer(qc, regs, cl_reg, site_idx, correct=True,
    selected_output=None):
    """
    Apply the parity stabilizer to the matter inside a site of the
    Hubbard defermoinised model, recording the result of the projective
    measurement on a classical register.

    Parameters
    ----------
    qc : QuantumCircuit
        hubbard quantum  circuit
    regs : dict
        Dictionary of the SiteRegisters
    cl_reg : str
        Name of the classical register where to store the measure
    site_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up
    correct : bool, optional
        If True, apply the correction after the stabilizer measurements.
        Otherwise do nothing. Default to True
    selected_output : int, optional
        A priori selected output for the projective measurement. If None,
        the measurement is draw according to a random number. Default to None.

    Return
    ------
    QuantumCircuit
        The quantum circuit after the application of the stabilizer

    Example
    -------
    We report here an example of the site numbering

    .. code-block::

        (0,2)--(1,2)--(2,2)
          |      |      |
        (0,1)--(1,1)--(2,1)
          |      |      |
        (0,0)--(1,0)--(2,0)

    """
    if not isinstance(qc, Qcircuit):
        raise TypeError(f'qc must be of type Qcircuit, not {type(qc)} when doing mps '+
            'simulations.')

    if not cl_reg in qc.cregisters:
        qc.add_cregister(cl_reg, 1)

    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]})' ]

    ancilla = str(uuid.uuid4())
    qc.add_qregister(ancilla, [site_reg['u']+1], reference_register='default')

    # Apply controlled x for checking the parity of a site
    for matter in ('u', 'd'):
        qc.cx( [site_reg[matter], 0], qreg = ['default', ancilla] )

    # Apply projective measurement
    qc.measure_projective(0, 0, ancilla, cl_reg, selected_output=selected_output)

    if correct:
        c_if = ClassicalCondition(cl_reg, 1, 0)
        # Apply a controlled x operation
        qc.x(site_reg['u'], c_if = c_if)

    # Reset the ancilla
    qc.remove_qregister(ancilla)

    return qc

def apply_link_parity_stabilizer(qc, regs, cl_reg, link_idx, correct=True,
    selected_output=None):
    """
    Apply the parity stabilizer to a link of the Hubbard defermoinised model,
    recording the result of the projective measurement on a classical
    register

    Parameters
    ----------
    qc : QuantumCircuit
        hubbard quantum  circuit
    regs : dict
        Dictionary of the SiteRegisters
    cl_reg : str
        Name of the classical register where to store the measure
    link_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up.
        Please refer to the example: in this case the numbering is NOT
        trivial.
    correct : bool, optional
        If True, apply the correction after the stabilizer measurements.
        Otherwise do nothing. Default to True
    selected_output : int, optional
        A priori selected output for the projective measurement. If None,
        the measurement is draw according to a random number. Default to None.

    Return
    ------
    QuantumCircuit
        The quantum circuit after the application of the stabilizer

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
    if not isinstance(qc, Qcircuit):
        raise TypeError(f'qc must be of type Qcircuit, not {type(qc)} when doing mps '+
            'simulations.')

    if not cl_reg in qc.cregisters:
        qc.add_cregister(cl_reg, 1)

    # If the y component of the index is even
    # the link is horizontal, otherwise vertical
    is_horizontal = (link_idx[1]%2 == 0)
    if is_horizontal:
        involved_sites = [
            f'q({link_idx[0]}, {link_idx[1]})',
            f'q({link_idx[0]+1}, {link_idx[1]})'
        ]
        involved_rishons = ['e', 'w']
    else:
        involved_sites = [
            f'q({link_idx[0]}, {link_idx[1]-1})',
            f'q({link_idx[0]}, {link_idx[1]})'
        ]
        involved_rishons = ['n', 's']

    # Add ancilla
    first_site = regs[involved_sites[0]][involved_rishons[0]]
    second_site = regs[involved_sites[1]][involved_rishons[1]]
    ancilla = str(uuid.uuid4())
    qc.add_qregister(ancilla, [first_site+1], reference_register='default')

    # Apply controlled x for checking the parity of a site
    qc.cx([first_site, 0], ['default', ancilla])
    if np.abs(first_site-second_site)>1:
        qc.move_site(0, ancilla, second_site)
    qc.cx([second_site, 0], ['default', ancilla])

    # Apply projective measurement
    qc.measure_projective(0, 0, ancilla, cl_reg, selected_output=selected_output)

    if correct:
        c_if = ClassicalCondition(cl_reg, value=1, idx=0)
        # Apply a controlled x operation
        qc.x(first_site, c_if = c_if)

    # Reset the ancilla
    qc.remove_qregister(ancilla)

    return qc