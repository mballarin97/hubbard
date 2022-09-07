from qiskit.circuit import Parameter
import hubbard as hbb
from hubbard.evolution import generate_global_hopping, generate_global_onsite, generate_hopping
from qiskit import AncillaRegister, ClassicalRegister, execute, transpile
from qiskit.providers.aer import StatevectorSimulator

if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='double')

    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancilla = ClassicalRegister(1, 'ca0')
    cancilla1 = ClassicalRegister(1, 'ca1')

    # Initialize Hubbard state
    shape = (2, 2)
    regs, qc = hbb.hubbard_circuit(shape, qancilla, [cancilla, cancilla1] )
    qc = hbb.initialize_chessboard(qc, regs)
    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancilla, (0,0) )
    qc.barrier()
    print(qc)

    print(regs['q(1, 0)']['w'] )

    hop = generate_hopping(regs, 'lh0', 'u')
    print(hop)
    global_hop = generate_global_hopping(qc, regs, 'lh0', 'u')
    global_onsite = generate_global_onsite(qc, regs, (0, 0))

    keys = list(global_hop.keys())[::-1]
    hopping = keys[0][::-1]
    os_keys = list(global_onsite.keys())[::-1]
    onsite = os_keys[0][::-1]
    for idx, qq in enumerate(qc.qubits):
        print(qq, hopping[idx], onsite[idx])

    res = execute(qc, backend=backend )
    results = res.result()
    statevect = results.get_statevector(qc)

    state_str = hbb.lattice_str(qc, statevect.data, regs, shape)

    print(state_str)


    evol = hbb.evolution_operation(qc, regs, shape, 1j,-8, Parameter('dt'), 1) # Parameter('J'), Parameter('U'), Parameter('dt'), 1)
    qc.append(evol, range(qc.num_qubits))
    print(transpile( qc, basis_gates=['rx', 'ry', 'rz', 'h', 'cx']) )
