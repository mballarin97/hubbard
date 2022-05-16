from qcomps.circuit import Qcircuit
from .registers import QCSiteRegister

def hubbard_circuit(shape, mapping_1d='quasi_1d_chain'):
    """
    Initialize a quantum circuit with the Hubbard
    shape. All the hubbard-related sites are stored
    in the quantum register 'default'. This is due to
    the added complexity of taking care of the linearization.

    Parameters
    ----------
    shape : tuple
        Shape of the 2d Hubbard system
    mapping_1d : str, optional
        Way of mapping the 2d hubbard system to a 1-dimensional
        system. Default to 'quasi_1d_chain'.

    Return
    ------
    dict
        Dictionary of the registers, where the key is the position
        of the vertex and the value the SiteRegister
    QuantumCircuit
        The quantum circuit
    """

    registers = {}
    num_sites = 0
    for ii in range(shape[0]):
        for jj in range(shape[1]):
            reg = QCSiteRegister(ii, jj, shape, mapping_1d)
            registers[reg.name] = reg
            num_sites += len(reg)
    qc = Qcircuit(num_sites)

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
    qc : Qcircuit
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
            qc.x(reg['u'], qreg='default')
            qc.x(reg['d'], qreg='default')

    if final_barrier:
        qc.barrier()

    return qc
