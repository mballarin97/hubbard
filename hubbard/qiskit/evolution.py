# code from https://github.com/DavitKhach/quantum-algorithms-tutorials/blob/master/Hamiltonian_simulation.ipynb

from copy import deepcopy
from qiskit import *
from qiskit.aqua.operators import WeightedPauliOperator
import numpy as np
from ..operators import generate_hopping
from qiskit.extensions.quantum_initializer.initializer import initialize
from qiskit.providers.aer import StatevectorSimulator

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

def generate_global_hopping(qc, regs, link_idx, species, coupling=1):
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
    coupling : float, optional
        Coupling in the link

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
    values = coupling * np.array(list(local_operators.values()))
    return dict(zip(global_operator, values) )

def generate_global_onsite(qc, regs, site_idx, potential=1):
    """
    Generate the onsite operators of the hamiltonian given the
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
    potential : 1
        Value of the coefficient of the potential

    Example
    -------
    We report here an example of the vertex numbering

    .. code-block::

        (0,2)--(1,2)--(2,2)
          |      |      |
        (0,1)--(1,1)--(2,1)
          |      |      |
        (0,0)--(1,0)--(2,0)
    """
    # Generate the local operator, defined only on the interested
    # registers
    site_reg = regs[ f'q({site_idx[0]}, {site_idx[1]})' ]
    num_qubs = qc.num_qubits-1

    # Generate the global operators, padded with identities
    global_operator = ['I']*num_qubs
    operators =  'ZZ'+ 'I'*(len(site_reg.map)-2)
    for ii, qubit in enumerate(site_reg.qregister):
        qidx = qc.find_bit(qubit).index
        global_operator[qidx] = operators[ii]

    global_operator = ''.join(global_operator)
    return {global_operator: 0.5*potential}

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

def evolution_operation(qc, regs, shape,
    interaction_constant, onsite_constant, dt, num_timesteps):
    """
    Generate the evolution istruction for the Hubbard model

    Parameters
    ----------
    qc : QuantumCircuit
        qiskit hubbard quantum circuit
    regs : dict
        Dictionary of the registers
    shape : tuple
        Shape of the lattice
    interaction_constant : float
        Value of the interaction constant
    onsite_constant : float
        Value of the on-site constant
    dt : float
        Timestep
    num_timesteps : int
        Number of timesteps

    Returns
    -------
    instruction
        qinstruction with the evolution
    """

    # Links available in lattice of given shape
    avail_links = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]+1)]
    avail_links += [(shape[0]-1, jj) for jj in range(shape[1]) if jj%2==1]

    hubbard_hamiltonian = {}
    # Generate hopping term
    for link_idx in avail_links:
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hubbard_hamiltonian.update(hop_term)
    # Generate on-site term
    sites = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    for site in sites:
        onsite_term = generate_global_onsite(qc, regs, site, onsite_constant)
        hubbard_hamiltonian.update(onsite_term)

    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Create evolution instruction
    evolution_instruction = hamiltonian.evolve_instruction(evo_time=dt,
        expansion_order=2, num_time_slices=num_timesteps)

    return evolution_instruction

def compute_expectation(qc, regs, shape, statevect, interaction_constant):

    backend = StatevectorSimulator(precision='double')
    # Links available in lattice of given shape
    avail_links = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]+1)]
    avail_links += [(shape[0]-1, jj) for jj in range(shape[1]) if jj%2==1]

    hubbard_hamiltonian = {}
    for link_idx in avail_links:
        for specie in ('u', 'd'):
            hop_term = generate_global_hopping(qc, regs, link_idx, specie, interaction_constant)
            hubbard_hamiltonian.update(hop_term)

    pauli_dict = from_operators_to_pauli_dict(hubbard_hamiltonian)
    hamiltonian = WeightedPauliOperator.from_dict(pauli_dict)

    # Remove the ancilla qubit
    dense_state = statevect.data.reshape([2]*qc.num_qubits)
    dense_state = np.tensordot(dense_state, np.ones(2), ([0], [0]))
    statevect._data = dense_state.reshape(2**(qc.num_qubits-1) )
    hamiltonian_expectation = hamiltonian.evaluate_with_statevector(statevect)

    if False:
        qc = QuantumCircuit(*qc.qregs)
        initialize(qc, statevect)
        for key, val in hubbard_hamiltonian.items():
            current_qc = deepcopy(qc)
            for idx, pauli in enumerate(key):
                if pauli == 'X':
                    current_qc.x(idx)
                elif pauli == 'Z':
                    current_qc.z(idx)
                elif pauli == 'Y':
                    current_qc.y(idx)

            res = execute(qc, backend=backend )
            results = res.result()
            ket = results.get_statevector(qc)
            hamiltonian_expectation += np.vdot(statevect, ket)*val

    return hamiltonian_expectation
