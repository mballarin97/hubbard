import numpy as np

def apply_plaquette_stabilizers(qc, regs, ancilla, cl_reg, plaquette_idx):
    """
    Apply the stabilizer to a plaquette of the Hubbard defermoinised model,
    recording the result of the projective measurement on a classical
    register

    Parameters
    ----------
    qc : QuantumCircuit
        hubbard quantum  circuit
    regs : dict
        Dictionary of the SiteRegisters
    ancilla : QuantumRegister
        ancilla register for measurement
    cl_reg : ClassicalRegister
        Classical register index where to store the measurement
    plaquette_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up

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
    # Keys are created as:
    # - bottom left
    # - upper left
    # - bottom right
    # - upper right
    corner_order = ['bl', 'ul', 'br', 'ur']
    # Relative rishons contains the rishons used in the stabilizer
    # based on the corner of the site
    relative_rishons = {
        'bl' : ['n', 'e'],
        'ul' : ['s', 'e'],
        'br' : ['n', 'w'],
        'ur' : ['s', 'w'],
    }
    # Site registers that forms the plaquette
    involved_regs = [ f'q({plaquette_idx[0]+ii}, {plaquette_idx[1]+jj})'
        for ii in range(2) for jj in range(2)]
    involved_regs = dict(zip(corner_order, involved_regs))

    # Apply hadamard to the ancilla
    qc.h( ancilla )

    # Apply cx
    for corner in corner_order:
        for rishon in relative_rishons[corner]:
            qc.cx( ancilla, regs[involved_regs[corner]][rishon] )

    # Measure ancilla on x, i.e. hadamard+measure on z
    qc.h(ancilla)
    qc.measure(ancilla, cl_reg[0])

    # Apply a controlled z operation
    qc.z(regs[involved_regs['br']]['n']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc


def apply_site_parity_stabilizer(qc, regs, ancilla, cl_reg, site_idx):
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
    ancilla : QuantumRegister
        ancilla register for measurement
    cl_reg : ClassicalRegister
        Classical register index where to store the measurement
    site_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up

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
    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]}' ]

    # Apply controlled x for checking the parity of a site
    for matter in ('u', 'd'):
        qc.cx(site_reg[matter], matter)

    # Apply projective measurement
    qc.measure(ancilla, cl_reg)

    # Apply a controlled x operation
    qc.x(site_reg['u']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc

def apply_link_parity_stabilizer(qc, regs, ancilla, cl_reg, link_idx):
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
    ancilla : QuantumRegister
        ancilla register for measurement
    cl_reg : ClassicalRegister
        Classical register index where to store the measurement
    link_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up.
        Please refer to the example: in this case the numbering is NOT
        trivial.

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

    # Apply controlled x for checking the parity of a site
    for site, rishon in zip(involved_sites, involved_rishons):
        qc.cx(regs[site][rishon], ancilla)

    # Apply projective measurement
    qc.measure(ancilla, cl_reg)

    # Apply a controlled x operation
    qc.x(regs[involved_sites[0]][involved_rishons[0]]).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc