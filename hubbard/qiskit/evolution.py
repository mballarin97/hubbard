# code from https://github.com/DavitKhach/quantum-algorithms-tutorials/blob/master/Hamiltonian_simulation.ipynb

from qiskit import *
import numpy as np
from ..operators import generate_hopping

def exp_all_z(circuit, quantum_register,
              pauli_idexes, control_qubit=None, t=1):
    """
    The implementation of exp(iZZ..Z t), where Z is
    the Pauli Z operator, t is a parameter.
    :param circuit: QuantumCircuit.
    :param quantum_register: QuantumRegister.
    :param pauli_idexes: the indexes from quantum_register that
                         correspond to entries not equal to I:
                         e.g. if we have XIYZI then the
                         pauli_idexes = [0,2,3].
    :param control_qubit: the control Qubit from QuantumRegister
                          other than quantum_register.
    :param t: the parameter t in exp(iZZ..Z t).
    """
    # the controlled_exp(iIt) special case
    if len(pauli_idexes) == 0 and control_qubit is not None:
        circuit.add_register(control_qubit.register)
        circuit.u1(t, control_qubit)
        return

    # the first CNOTs
    for i in range(len(pauli_idexes) - 1):
        circuit.cx(quantum_register[pauli_idexes[i]],
                   quantum_register[pauli_idexes[i + 1]])

    # Rz gate
    if control_qubit is None:
        circuit.rz(-2 * t, quantum_register[pauli_idexes[-1]])
    else:
        circuit.add_register(control_qubit.register)
        circuit.crz(-2 * t,
                    control_qubit, quantum_register[pauli_idexes[-1]])

    # the second CNOTs
    for i in reversed(range(len(pauli_idexes) - 1)):
        circuit.cx(quantum_register[pauli_idexes[i]],
                   quantum_register[pauli_idexes[i + 1]])

def exp_pauli(pauli, quantum_register, control_qubit=None, t=1):
    """
    The circuit for the exp(i P t), where P is the Pauli term,
    t is the parameter.
    :param pauli: the string for the Pauli term: e.g. "XIXY".
    :param quantum_register: QuantumRegister.
    :param control_qubit: the control Qubit from QuantumRegister
                          other than quantum_register.
    :param t: the parameter t in exp(i P t).
    :return: QuantumCircuit that implements exp(i P t) or
             control version of it.
    """
    if len(pauli) != len(quantum_register):
        raise Exception("Pauli string doesn't match to the quantum register")

    pauli_circuit = QuantumCircuit(quantum_register)
    circuit_bracket = QuantumCircuit(quantum_register)
    pauli_idexes = []

    for i in range(len(quantum_register)):
        if pauli[i] == 'I':
            continue
        elif pauli[i] == 'Z':
            pauli_idexes.append(i)
        elif pauli[i] == 'X':
            circuit_bracket.h(quantum_register[i])
            pauli_idexes.append(i)
        elif pauli[i] == 'Y':
            circuit_bracket.u2(np.pi / 2, np.pi / 2, quantum_register[i])
            pauli_idexes.append(i)

    pauli_circuit += circuit_bracket
    exp_all_z(pauli_circuit, quantum_register, pauli_idexes, control_qubit, t)
    pauli_circuit += circuit_bracket

    return pauli_circuit

def hamiltonian_simulation(hamiltonian, quantum_register=None,
                           control_qubit=None, t=1, trotter_number=1):
    """
    The implementation of exp(iHt), where H is the Hamiltonian
    operator, t is the parameter.
    :param hamiltonian: dictionary of Pauli terms with their weights:
                        e.g. {"XZX": 2, "ZYI": 5, "IYZ": 7}.
    :param quantum_register: QuantumRegister.
    :param control_qubit: the control Qubit from QuantumRegister
                          other than quantum_register.
    :param t: the parameter t in exp(iHt).
    :param trotter_number: the Trotter number.
    :return: QuantumCircuit that corresponds to exp(iHt)
             or control version of it
    """
    if quantum_register is None:
        quantum_register = QuantumRegister(len(list(hamiltonian.keys())[0]))
    if control_qubit in quantum_register:
        raise Exception("the control qubit is in the target register")

    delta_t = t / trotter_number
    exp_hamiltonian = QuantumCircuit(quantum_register)
    exp_delta_t = QuantumCircuit(quantum_register)

    for pauli in hamiltonian:
        weight = hamiltonian[pauli]
        exp_delta_t += exp_pauli(pauli, quantum_register,
                                 control_qubit, weight * delta_t)

    for i in range(trotter_number):
        exp_hamiltonian += exp_delta_t

    return exp_hamiltonian

def generate_global_hopping(qc, regs, link_idx, species):
    """
    Generate the hopping operators of the hamiltonian given the
    jordan-wigner transformation. Thus, the output are Pauli strings.
    The operator is global in the sense that is padded with identities.

    Parameters
    ----------
    qc : :py:class:`QuantumCircuit`
        Quantum circuit class containing the Hubbard circuit
    regs : dict
        Dictionary of the SiteRegisters
    link_idx : tuple
        Unique identifier of the link where the hopping will take place
    species : str
        Matter species involved in the hopping

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
    # Generate the local operator, defined only on the interested
    # registers
    local_operators, (from_site_reg, to_site_reg) = generate_hopping(regs, link_idx, species)
    operators = local_operators.keys()
    num_qubs = np.sum([len(reg.map) for reg in regs.values()])

    # Generate the global operators, padded with identities
    global_operator = [ ['I']*num_qubs, ['I']*num_qubs ]
    operators = [op.split('⊗') for op in operators]
    for ii, qubit in enumerate(from_site_reg.qregister):
        qidx = qc.find_bit(qubit).index
        for jj in range(2):
            global_operator[jj][qidx] = operators[jj][0][ii]

    for ii, qubit in enumerate(to_site_reg.qregister):
        qidx = qc.find_bit(qubit).index
        for jj in range(2):
            global_operator[jj][qidx] = operators[jj][1][ii]

    global_operator = [''.join(gl) for gl in global_operator]
    return dict(zip(global_operator, local_operators.values()) )

def from_operators_to_pauli_dict(pauli_hamiltonian):
    """
    Transform a Hamiltonian described as a dict, where the keys
    are the pauli strings and the values the coefficients into
    a pauli dict that can be read by the qiskit class
    :py:class:`WeightedPauliOperator`.

    Parameters
    ----------
    pauli_hamiltonian : dict
        Pauli hamiltonian. the keys are the pauli strings and
        the values the coefficients

    Returns
    -------
    dict
        Pauli dict to be used in `WeightedPauliOperator.from_dict`
    """
    paulis = []
    for label, coeff in pauli_hamiltonian.items():
        paulis += [
            {
                "coeff": {"imag": np.imag(coeff), "real": np.real(coeff)},
                "label": label
            }
        ]

    return { 'paulis': paulis }