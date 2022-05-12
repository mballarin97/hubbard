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
    ancilla_register : QuantumRegister
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