from qiskit import QuantumCircuit
from .registers import SiteRegister

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

    registers = {}
    reg_for_init = []
    for ii in range(shape[0]):
        for jj in range(shape[1]):
            reg = SiteRegister(ii, jj, shape)
            registers[reg.name] = reg
            reg_for_init.append( reg.qregister)

    qc = QuantumCircuit(*reg_for_init, ancilla_register, classical_registers,
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
