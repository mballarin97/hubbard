
def apply_plaquette_stabilizers(qc, regs, ancilla, cl_reg, plaquette_idx, correct=True):
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
    correct : bool, optional
        If True, apply the correction after the stabilizer measurements.
        Otherwise do nothing. Default to True

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
        'ul' : ['e'],
        'br' : ['n'],
        'ur' : [],
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
            print(corner, rishon, regs[involved_regs[corner]][rishon] )
            qc.cx( ancilla, regs[involved_regs[corner]][rishon] )

    # Measure ancilla on x, i.e. hadamard+measure on z
    qc.h(ancilla)
    qc.measure(ancilla, cl_reg[0])

    if correct:
        # Apply a controlled z operation
        qc.z(regs[involved_regs['br']]['n']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc


def apply_vertex_parity_stabilizer(qc, regs, ancilla, cl_reg, site_idx, correct=True):
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
    correct : bool, optional
        If True, apply the correction after the stabilizer measurements.
        Otherwise do nothing. Default to True

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
    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]})' ]

    # Apply controlled x for checking the parity of a site
    for matter in ('u', 'd'):
        qc.cx(site_reg[matter], ancilla)

    # Apply projective measurement
    qc.measure(ancilla, cl_reg)

    if correct:
        # Apply a controlled x operation
        qc.x(site_reg['u']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc
