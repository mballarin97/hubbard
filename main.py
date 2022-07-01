# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from tqdm import tqdm
import os

from qiskit.extensions.quantum_initializer.initializer import initialize
from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator

import hubbard as hbb
import hubbard.observables as obs


if __name__ == '__main__':
    # Initialize parser
    parser = hbb.hubbard_parser()
    args = parser.parse_args()

    # ============= Initialize parameters of the simulation =============
    # Shape of the lattice
    shape = args.shape
    # Timestep of the evolution
    time_step = args.dt
    # Number of trotterization steps for the single timestep
    num_trotter_steps = args.num_trotter_steps
    # Hopping constant, usually called J or t
    hopping_constant = args.t
    # Onsite constant, usually called U
    if isinstance(args.U, list):
        onsite_constant = np.linspace(args.U[0], args.U[1], args.U[2])
    elif isinstance(args.U, str):
        onsite_constant = np.loadtxt(args.U)
    # Number of steps in the evolution
    evolution_steps = len(onsite_constant)
    parameters_dict = vars(args)
    parameters_dict['U'] = onsite_constant
    # Plaquettes definition
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]

    # ============= Initialize results directory =============
    data_folder = 'data'
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    subfolders = [ int(f.name) for f in os.scandir(data_folder) if f.is_dir() ] + [-1]
    dir_name = os.path.join('data', str(max(subfolders)+1) )
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    with open(os.path.join(dir_name, 'params.json'), 'w') as fh:
        hbb.write_json(parameters_dict, fh)

    # ============= Initialize qiskit variables =============
    backend = StatevectorSimulator(precision='double')
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes))]

    # ============= Initialize Hubbard circuit =============
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )
    qc = hbb.initialize_chessboard(qc, regs)
    for ii, pp in enumerate(plaquettes):
        qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )
    qc.barrier()

    # ============= Apply Evolution =============
    kinetic_exps = np.zeros(evolution_steps)
    u_and_d_exps = np.zeros((evolution_steps, 2*len(regs)) )
    ud_exps = np.zeros((evolution_steps, len(regs)) )
    entanglement_exps = np.zeros(evolution_steps)
    symmetry_check = np.zeros((evolution_steps, len(plaquettes)), dtype=int )
    idx = 0
    for site_const in tqdm(onsite_constant):
        # Start from the state at the previous timestep
        if idx > 0:
            qc = QuantumCircuit(*qc.qregs, *qc.cregs)
            init_func = initialize(qc, statevect, range(qc.num_qubits))

        # Create evolution circuit
        evolution_instruction = hbb.evolution_operation(qc, regs, shape,
            hopping_constant, site_const, time_step, num_trotter_steps)
        qc.append(evolution_instruction, range(qc.num_qubits))
        # Apply plaquette stabilizer to check if we stay in the right symmetry sector
        #for ii, pp in enumerate(plaquettes):
        #    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )

        # Simulate the circuit
        res = execute(qc, backend=backend )
        results = res.result()
        statevect = results.get_statevector(qc)
        counts = results.get_counts()

        symmetry_check[idx, :] = list( list(counts.keys())[0] )
        kinetic_exps[idx] = obs.compute_kinetic_expectation(qc, regs, shape, statevect, 1)
        u_and_d_exps[idx, :] = obs.compute_up_and_down_expectation(qc, regs, statevect)
        ud_exps[idx, :] = obs.compute_updown_expectation(qc, regs, statevect)
        entanglement_exps[idx] = obs.compute_entanglement(qc, regs, shape, statevect)

        idx += 1
        np.savetxt(os.path.join(dir_name, 'symmetry_check.txt'), symmetry_check[:idx], fmt='%d')
        np.savetxt(os.path.join(dir_name, 'kinetic.txt'), kinetic_exps[:idx])
        np.savetxt(os.path.join(dir_name, 'u_and_d.txt'), u_and_d_exps[:idx, :])
        np.savetxt(os.path.join(dir_name, 'ud.txt'), ud_exps[:idx, :])
        np.savetxt(os.path.join(dir_name, 'entanglement.txt'), entanglement_exps[:idx])
