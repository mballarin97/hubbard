from qiskit import QuantumCircuit
from .registers import HubbardRegister

def hubbard_circuit(shape, ancilla_register, classical_registers):
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

    Return
    ------
    dict
        Dictionary of the registers, where the key is the position
        of the vertex and the value the SiteRegister
    QuantumCircuit
        The quantum circuit
    """

    registers = HubbardRegister(shape)

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
    for reg in regs.registers.values():
        if reg.is_even:
            qc.x(reg['u'])
            qc.x(reg['d'])

    if final_barrier:
        qc.barrier()

    return qc
