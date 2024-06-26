# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

__all__ = ['apply_plaquette_stabilizers', 'apply_vertex_parity_stabilizer', "plaquette_operator", "vertex_operator"]

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
        'bl' : [],
        'ul' : ['s', 'e'],
        'br' : ['n', 'w'],
        'ur' : [],
    }
    # Site registers that forms the plaquette
    involved_regs = [ f'q({plaquette_idx[0]+ii}, {plaquette_idx[1]+jj})'
        for ii in range(2) for jj in range(2)]
    involved_regs = dict(zip(corner_order, involved_regs))

    # Apply hadamard to the ancilla
    qc.h( ancilla )

    # Apply cx and cy
    for corner in corner_order:
        for rishon in relative_rishons[corner]:
            if regs[involved_regs["bl"]].is_even:
                if corner == 'br':
                    qc.cx( ancilla, regs[involved_regs[corner]][rishon] )
                elif corner == 'ul':
                    qc.cy( ancilla, regs[involved_regs[corner]][rishon] )
            else:
                if corner == 'br':
                    qc.cy( ancilla, regs[involved_regs[corner]][rishon] )
                elif corner == 'ul':
                    qc.cx( ancilla, regs[involved_regs[corner]][rishon] )

    # Apply cz if qubits are available
    if regs[involved_regs["bl"]].is_even:
        for rishon in ('n', 'w'):
            if rishon in regs[involved_regs['ul']]._keys:
                qc.cz( ancilla, regs[involved_regs['ul']][rishon] )
    else:
        for rishon in ('s', 'e'):
            if rishon in regs[involved_regs['br']]._keys:
                qc.cz( ancilla, regs[involved_regs['br']][rishon] )

    # Measure ancilla on x, i.e. hadamard+measure on z
    qc.h(ancilla)
    qc.measure(ancilla, cl_reg[0])

    if correct:
        # Apply a controlled z operation
        print('correcting', regs[involved_regs['br']]['n'])
        qc.z(regs[involved_regs['br']]['n']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc


def apply_vertex_parity_stabilizer(qc, regs, ancilla, cl_reg, site_idx, correct=True):
    """
    Apply the parity stabilizer to the matter and the link inside a site of the
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
    for matter in site_reg._keys:
        qc.cx(site_reg[matter], ancilla)

    # Apply projective measurement
    qc.measure(ancilla, cl_reg)

    if correct:
        # Apply a controlled x operation
        if "e" in site_reg._keys:
            qc.x(site_reg['e']).c_if(cl_reg, 1)
        elif "n" in site_reg._keys:
            qc.x(site_reg['n']).c_if(cl_reg, 1)
        else:
            qc.x(site_reg['s']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    return qc


def plaquette_operator(qc, regs, plaquette_idx):
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
    plaquette_idx : tuple of ints
        XY position of the plaquette where you will apply the stabilizar.
        Plaquettes are numbered from 0, left to right, from low to up

    Return
    ------
    str
        String of the plaquette operator

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
        'bl' : [],
        'ul' : ['s', 'e'],
        'br' : ['n', 'w'],
        'ur' : [],
    }
    # Site registers that forms the plaquette
    involved_regs = [ f'q({plaquette_idx[0]+ii}, {plaquette_idx[1]+jj})'
        for ii in range(2) for jj in range(2)]
    involved_regs = dict(zip(corner_order, involved_regs))

    # Apply hadamard to the ancilla
    operator = list('I'*qc.num_qubits)
    # Apply cx and cy
    for corner in corner_order:
        for rishon in relative_rishons[corner]:
            qubit = regs[involved_regs[corner]][rishon]
            qidx = qc.find_bit(qubit).index
            if regs[involved_regs["bl"]].is_even:
                if corner == 'br':
                    operator[qidx] = "X"
                elif corner == 'ul':
                    operator[qidx] = "Y"
            else:
                if corner == 'br':
                    operator[qidx] = "Y"
                elif corner == 'ul':
                    operator[qidx] = "X"

    # Apply cz if qubits are available
    if regs[involved_regs["bl"]].is_even:
        for rishon in ('n', 'w'):
            if rishon in regs[involved_regs['ul']]._keys:
                qubit = regs[involved_regs['ul']][rishon]
                qidx = qc.find_bit(qubit).index
                operator[qidx] = "Z"
    else:
        for rishon in ('s', 'e'):
            if rishon in regs[involved_regs['br']]._keys:
                qubit = regs[involved_regs['br']][rishon]
                qidx = qc.find_bit(qubit).index
                operator[qidx] = "Z"

    return "".join(operator[::-1])

def vertex_operator(qc, regs, site_idx):
    """
    Generate the stabilizer operator of the Hubbard defermoinised model,
    recording the result of the projective measurement on a classical
    register

    Parameters
    ----------
    qc : QuantumCircuit
        hubbard quantum  circuit
    regs : dict
        Dictionary of the SiteRegisters
    site_idx : tuple of ints
        XY position of the vertex_idx where you will apply the stabilizar.

    Return
    ------
    str
        String of the plaquette operator

    """
    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]})' ]
    names = ["u", "d", "e", "w", "n", "s"]

    # Apply hadamard to the ancilla
    operator = list('I'*qc.num_qubits)
    # Apply zz
    for name in names:
        if name in site_reg._keys:
            qubit = site_reg[name]
            qidx = qc.find_bit(qubit).index
            operator[qidx] = "Z"

    return "".join(operator[::-1])
