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

__all__ = ['hubbard_circuit', 'initialize_chessboard', 'initialize_superposition_chessboard']

def hubbard_circuit(shape, ancilla_register, classical_registers, ordering=None):
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

    registers = HubbardRegister(shape, ordering)

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