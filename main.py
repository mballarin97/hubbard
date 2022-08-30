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
from shutil import rmtree

from qiskit.extensions.quantum_initializer.initializer import initialize
from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator

import hubbard as hbb
import hubbard.observables as obs


if __name__ == '__main__':
    # Initialize parser
    parser = hbb.hubbard_parser()
    args = parser.parse_args()

    if args.clear:
        rmtree('data')
        exit(0)

    # ============= Initialize parameters of the simulation =============
    # Shape of the lattice
    shape = (int(args.shape[0]), int(args.shape[1]) )
    # Timestep of the evolution
    time_step = float(args.dt)
    # Number of trotterization steps for the single timestep
    num_trotter_steps = int(args.num_trotter_steps)
    # Hopping constant, usually called J or t
    hopping_constant = float(args.t)
    # Onsite constant, usually called U
    if not args.Ustep:
        onsite_constant = np.linspace(float(args.Umin), float(args.Umax), int(args.num_timesteps))
    else:
        first_evol = int(int(args.num_timesteps)/10)
        onsite_constant = np.array( [float(args.Umin)]*first_evol +
                            [float(args.Umax)]*(int(args.num_timesteps)-first_evol) )
    # Number of steps in the evolution
    evolution_steps = int(args.num_timesteps)
    parameters_dict = vars(args)
    parameters_dict.pop('clear')
    parameters_dict['U'] = onsite_constant
    # If True, apply stabilizers at each timestep
    apply_stabilizers = False

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
    entanglement_half_exps = np.zeros(evolution_steps)
    entanglement_matter_links_exps = np.zeros(evolution_steps)
    symmetry_check = np.zeros((evolution_steps, len(plaquettes)), dtype=int )
    idx = 0

    # Add a first timestep with very strong U, the initial state setting
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
        if apply_stabilizers:
            for ii, pp in enumerate(plaquettes):
                qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )

        # Simulate the circuit
        res = execute(qc, backend=backend )
        results = res.result()
        statevect = results.get_statevector(qc)
        if apply_stabilizers:
            counts = results.get_counts()

        if apply_stabilizers:
            symmetry_check[idx, :] = list( list(counts.keys())[0] )
        kinetic_exps[idx] = obs.compute_kinetic_expectation(qc, regs, shape, statevect, 1)
        u_and_d_exps[idx, :] = obs.compute_up_and_down_expectation(qc, regs, statevect)
        ud_exps[idx, :] = obs.compute_updown_expectation(qc, regs, statevect)
        entanglement_half_exps[idx] = obs.compute_half_entanglement(qc, regs, shape, statevect)
        entanglement_matter_links_exps[idx] = obs.compute_links_matter_entanglement(qc, regs, shape, statevect)

        idx += 1
        np.savetxt(os.path.join(dir_name, 'symmetry_check.txt'), symmetry_check[:idx], fmt='%d')
        np.savetxt(os.path.join(dir_name, 'kinetic.txt'), kinetic_exps[:idx])
        np.savetxt(os.path.join(dir_name, 'u_and_d.txt'), u_and_d_exps[:idx, :])
        np.savetxt(os.path.join(dir_name, 'ud.txt'), ud_exps[:idx, :])
        np.savetxt(os.path.join(dir_name, 'entanglement_half.txt'), entanglement_half_exps[:idx])
        np.savetxt(os.path.join(dir_name, 'entanglement_matter_link.txt'), entanglement_matter_links_exps[:idx])
