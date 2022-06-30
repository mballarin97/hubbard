import numpy as np

from qiskit.extensions.quantum_initializer.initializer import initialize
from qiskit import QuantumCircuit
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator

from hubbard.new_encoding.circuit import hubbard_circuit, initialize_chessboard
from hubbard.new_encoding.stabilizers import apply_plaquette_stabilizers
from hubbard.new_encoding.evolution import evolution_operation, compute_kinetic_expectation
from hubbard.utils import lattice_str
from tqdm import tqdm
from qmatchatea import print_state


if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='double')

    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancilla = ClassicalRegister(1, 'ca0')
    cancilla1 = ClassicalRegister(1, 'ca1')

    # Initialize Hubbard state
    quench = True
    shape = (2, 2)
    regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
    qc = initialize_chessboard(qc, regs)
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]
    for pp in plaquettes:
        qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, pp )
    qc.barrier()

    num_steps = 100 # Number of steps to pass from one onsite const to the other
    if quench:
        onsite_consts = [-8 for _ in range(num_steps//10)]+ [-1/8 for _ in range(num_steps-num_steps//10)]
    else:
        onsite_consts = np.linspace(-8, -1/8, num_steps)

    # Parameter for single step in the evolution
    int_const = 1
    dt = 0.1
    num_tsteps = 100

    expectations = []
    idx = 0
    for site_const in tqdm(onsite_consts):
        # Create evolution circuit
        evolution_instruction = evolution_operation(qc, regs, shape, int_const, site_const, dt, num_tsteps)
        if idx > 0:
            qc = QuantumCircuit(*qc.qregs)
            init_func = initialize(qc, statevect, range(qc.num_qubits))

        qc.append(evolution_instruction, range(qc.num_qubits))

        # Simulate the circuit
        res = execute(qc, backend=backend )
        results = res.result()
        statevect = results.get_statevector(qc)
        print_state(statevect.data)
        #import matplotlib.pyplot as plt
        #plt.plot(np.conj(statevect.data[statevect.data <0.1] )* statevect.data[statevect.data <0.1], 'o')
        #plt.show()
        exp = compute_kinetic_expectation(qc, regs, shape, statevect, 1)

        expectations.append(exp)
        idx += 1

    #res = lattice_str(statevect, regs, shape )

    #print( res )
        if quench:
            np.savetxt(f'results_quench{shape}.txt', expectations)
        else:
            np.savetxt(f'results_adiabatic{shape}.txt', expectations)