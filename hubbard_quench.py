import numpy as np

from qiskit.extensions.quantum_initializer.initializer import initialize
from qiskit import QuantumCircuit
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator

from hubbard.qiskit.circuit import hubbard_circuit, initialize_chessboard
from hubbard.qiskit.stabilizers import apply_plaquette_stabilizers
from hubbard.qiskit.evolution import evolution_operation, compute_expectation
from hubbard.utils import lattice_str
from tqdm import tqdm


if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='double')

    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancilla = ClassicalRegister(1, 'ca0')
    cancilla1 = ClassicalRegister(1, 'ca1')

    # Initialize Hubbard state
    shape = (2, 2)
    regs, qc = hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
    qc = initialize_chessboard(qc, regs)
    qc = apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
    qc.barrier()

    num_steps = 10 # Number of steps to pass from one onsite const to the other
    onsite_consts = np.linspace(-8, -1/8, num_steps)

    # Parameter for single step in the evolution
    int_const = 1
    dt = 0.1
    num_tsteps = 1

    expectations = []
    idx = 0
    for site_const in tqdm(onsite_consts):
        # Create evolution circuit
        evolution_instruction = evolution_operation(qc, regs, shape, int_const, site_const, dt, num_tsteps)
        if idx > 0:
            qc = QuantumCircuit(*qc.qregs)
            init_func = initialize(qc, statevect, range(qc.num_qubits-1))

        qc.append(evolution_instruction, range(qc.num_qubits-1))

        # Simulate the circuit
        res = execute(qc, backend=backend )
        results = res.result()
        statevect = results.get_statevector(qc)
        exp = compute_expectation(qc, regs, shape, statevect, int_const)

        expectations.append(exp)
        idx += 1

    res = lattice_str(statevect, regs, shape )

    print( res )
    np.savetxt('quench_results.txt', expectations)