# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumCircuit
from .registers import HubbardRegister

__all__ = ['hubbard_circuit', 'initialize_chessboard', 'initialize_superposition_chessboard',
            'initialize_repulsive_rows']

def hubbard_circuit(shape, ancilla_register, classical_registers, ordering=None, extra_leg=False):
    """
    Initialize a quantum circuit with the Hubbard
    shape

    Parameters
    ----------
    shape : tuple
        Shape of the 2d Hubbard system
    ancilla_register : AncillaRegister
        Quantum register defining the ancilla qubits
    classical_registers : ClassicalRegister
        Classical register for the measurements
    ordering : list of ints
        New order of the sites to minimize the entanglement.
        Default is None, with the following ordering:
        - All the vertical links, left to right, down to up
        - All the horizontal links, left to right, down to up
        - All the sites, left to right, down to up

    Return
    ------
    dict
        Dictionary of the registers, where the key is the position
        of the vertex and the value the SiteRegister
    QuantumCircuit
        The quantum circuit
    """

    registers = HubbardRegister(shape, ordering, extra_leg)

    qc = QuantumCircuit(*registers.qregisters, ancilla_register, *classical_registers,
        name=f'Hubbard {shape}')

    return registers, qc

def initialize_chessboard(qc, regs, final_barrier=True):
    """
    Initialize the hubbard state with the chessboard

    .. code-block::

        2 - 0 - 2 - 0
        0 - 2 - 0 - 2
        2 - 0 - 2 - 0

    Parameters
    ----------
    qc : QuantumCircuit
        The Hubbard quantum circuit
    regs : dict
        The dictionary of the site registers
    final_barrier : bool, optional
        If True, put a barrier after the initialization.
        Default to True

    Returns
    -------
    QuantumCircuit
        The quantum circuit with the initialization
    """
    for reg in regs.values():
        if reg.is_even:
            qc.x(reg['u'])
            qc.x(reg['d'])

    if final_barrier:
        qc.barrier()

    return qc

def initialize_superposition_chessboard(qc, regs, ancilla, cl_reg, correct=True,
    final_barrier=True):
    """
    Initialize the hubbard state with the superposition chessboard

    .. code-block::

        2 - 0 - 2 - 0       0 - 2 - 0 - 2
        0 - 2 - 0 - 2   +   2 - 0 - 2 - 0
        2 - 0 - 2 - 0       0 - 2 - 0 - 2

    Parameters
    ----------
    qc : QuantumCircuit
        The Hubbard quantum circuit
    regs : dict
        The dictionary of the site registers
    final_barrier : bool, optional
        If True, put a barrier after the initialization.
        Default to True

    Returns
    -------
    QuantumCircuit
        The quantum circuit with the initialization
    """

    qc = initialize_chessboard(qc, regs, final_barrier)

    # Apply hadamard to the ancilla
    qc.h( ancilla )

    for reg in regs.values():
        qc.cx(ancilla, reg['u'])
        qc.cx(ancilla, reg['d'])

    # Measure ancilla on x, i.e. hadamard+measure on z
    qc.h(ancilla)
    qc.measure(ancilla, cl_reg[0])

    if correct:
        # Apply a controlled z operation
        qc.z(regs['q(0, 0)']['u']).c_if(cl_reg, 1)

    # Reset the ancilla
    qc.reset(ancilla)

    if final_barrier:
        qc.barrier()

    return qc

def initialize_repulsive_rows(qc, regs, ancilla, cl_reg, shape, filling="h"):
    """
    Initialize the hubbard state with two particles in each
    row, while respecting the stabilizers constraints

    .. code-block::

        10 -1- 01   01 -1- 10   00 -0- 00   00 -0- 00
        |       |   |       |   |       |   |       |
        0       0 + 0       0 + 0       0 + 0       0
        |       |   |       |   |       |   |       |
        00 -0- 00   00 -0- 00   10 -1- 01   01 -1- 10

    Parameters
    ----------
    qc : QuantumCircuit
        The Hubbard quantum circuit
    regs : dict
        The dictionary of the site registers
    ancilla : QuantumRegister
        ancilla register for measurement
    cl_reg : ClassicalRegister
        Classical register index where to store the measurement
    shape : tuple
        Shape of the matter lattice


    Returns
    -------
    QuantumCircuit
        The quantum circuit with the initialization
    """

    # Create upper row
    qc.h(regs[f'q(0, {shape[1]-1})']['u'])
    for ii in range(shape[0]):
        if (ii+shape[1]-1)%2 == 1:
            from_r, to_r = 'u', 'd'
        else:
            from_r, to_r = 'd', 'u'
        # Apply cx in the same site
        qc.cx(regs[f'q({ii}, {shape[1]-1})'][from_r],
                regs[f'q({ii}, {shape[1]-1})'][to_r])
        if ii == shape[0]-1:
            continue
        # Apply cx to next site
        qc.cx(regs[f'q({ii}, {shape[1]-1})'][to_r],
                regs[f'q({ii+1}, {shape[1]-1})'][to_r])

        if ii%2==0:
            qc.x(regs[f'q({ii}, {shape[1]-1})']['e'] )

    # Add particles to reach half filling
    if filling == "h":
        for jj in range(shape[1]-1, 0, -1):
            for ii in range(shape[0]):
                for mm in ('u', 'd'):
                    qc.cx(regs[f'q({ii}, {jj})'][mm],
                            regs[f'q({ii}, {jj-1})'][mm])

        for jj in range(shape[1]-1, -1, -1):
            for ii in range(shape[0]):
                if (ii+jj)%2 == 1:
                    qc.x(regs[f'q({ii}, {jj})']['u'] )
                else:
                    qc.x(regs[f'q({ii}, {jj})']['d'] )
    else:
        # flip the bits to have only one particle per site
        for ii in range(shape[0]):
            if (ii+shape[1]-1)%2 == 1:
                qc.x(regs[f'q({ii}, {shape[1]-1})']['u'] )
            else:
                qc.x(regs[f'q({ii}, {shape[1]-1})']['d'] )
    qc.barrier()

    # Complete with the other rows
    for jj in range(shape[1]-1, 0, -1):
        qc.h(ancilla[0])
        for ii in range(shape[0]):
            for mm in ('u', 'd'):
                qc.cswap(ancilla[0],
                            regs[f'q({ii}, {jj-1})'][mm],
                            regs[f'q({ii}, {jj})'][mm]
                        )
            if ii != shape[0]-1:
                qc.cswap(ancilla[0],
                            regs[f'q({ii}, {jj-1})']['e'],
                            regs[f'q({ii}, {jj})']['e']
                        )
        qc.h(ancilla[0])
        qc.measure(ancilla[0], cl_reg[0])
        qc.z(regs[f'q({ii}, {jj})']['u']).c_if( cl_reg, 1)
        qc.z(regs[f'q({ii}, {jj})']['d']).c_if( cl_reg, 1)
        qc.reset(ancilla[0])
        qc.barrier()

    return qc

def initialize_for_chemical_charge(qc, regs, final_barrier=True):
    """
    Initialize the hubbard state with the chessboard

    .. code-block::

        2 - 0 - 2 - 0
        0 - 2 - 0 - 2
        2 - 0 - 2 - 0

    Parameters
    ----------
    qc : QuantumCircuit
        The Hubbard quantum circuit
    regs : dict
        The dictionary of the site registers
    final_barrier : bool, optional
        If True, put a barrier after the initialization.
        Default to True

    Returns
    -------
    QuantumCircuit
        The quantum circuit with the initialization
    """
    for reg in regs.values():
        if reg.is_even:
            qc.x(reg['u'])
            qc.x(reg['d'])

    if final_barrier:
        qc.barrier()

    return qc
