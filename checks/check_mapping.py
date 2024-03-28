# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Check if the mapping is correct by plotting the state preparation and one
application of the trotterized hamiltonian
"""

from qiskit.circuit import Parameter
import hubbard as hbb
from hubbard.evolution import generate_global_hopping, generate_global_onsite
from hubbard.operators import generate_hopping
from qiskit import AncillaRegister, ClassicalRegister, execute, transpile
from qiskit_aer.backends import StatevectorSimulator

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

    hop = generate_hopping(regs, 'lh0', 'u')
    print("Hopping term is:", hop)
    print("")
    global_hop = generate_global_hopping(qc, regs, 'lh0', 'u')
    global_onsite = generate_global_onsite(qc, regs, (0, 0))

    keys = list(global_hop.keys())[::-1]
    hopping = keys[0][::-1]
    os_keys = list(global_onsite.keys())[::-1]
    onsite = os_keys[0][::-1]
    print("Hopping and onsite term for each qubit are:")
    for idx, qq in enumerate(qc.qubits):
        print(qq, hopping[idx], onsite[idx])
    print("")

    # Run the circuit
    res = execute(qc, backend=backend )
    results = res.result()
    statevect = results.get_statevector(qc)
    state_str = hbb.lattice_str(qc, statevect.data, regs, shape)

    print("The state after the initialization to chessboard is:")
    print(state_str)
    print("")

    # Initialize an evolution operator
    evol = hbb.evolution_operation(qc, regs, shape, 1, 0, Parameter('dt'), 1)
    qc.append(evol, range(qc.num_qubits))
    print(transpile( qc, basis_gates=['rx', 'ry', 'rz', 'h', 'cx']) )
