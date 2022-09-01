# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from copy import deepcopy
import numpy as np
from tqdm import tqdm
import os
from shutil import rmtree

from qiskit import QuantumCircuit, AncillaRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qmatchatea import run_simulation
import qmatchatea as qtea
from qmatchatea.preprocessing import _preprocess_qk
from qmatchatea.qk_utils import qk_transpilation_params
from tn_py_frontend.observables import TNObservables, TNObsBondEntropy, TNState2File
from tn_py_frontend.emulator import MPS

import hubbard as hbb
import hubbard.mps_observables as obs


if __name__ == '__main__':

    # ============= Initialize parameters of the simulation =============
    # Shape of the lattice
    shape = (2, 2)
    # Number of trotterization steps for the single timestep
    num_trotter_steps = 1
    # Hopping constant, usually called J or t
    hopping_constant = 1
    # Onsite constant, usually called U
    onsite_constant = -8
    # Chemical potential
    chemical_potential = 1
    # Number of steps in the evolution
    alpha_steps = 100000
    # Maximum bond dimension of the simulation
    max_bond_dim = 10000

    parameters_dict = {}
    parameters_dict['U'] = onsite_constant
    parameters_dict['t'] = hopping_constant
    parameters_dict['mu'] = chemical_potential
    parameters_dict['mps'] = True
    parameters_dict['chi'] = max_bond_dim
    parameters_dict['num_trotter_steps'] = num_trotter_steps
    parameters_dict['num_timesteps'] = alpha_steps
    parameters_dict['shape'] = shape
    parameters_dict['adiabatic'] = True
    parameters_dict['Ustep'] = False
    parameters_dict['dt'] = 1/alpha_steps
    conv_params = qtea.QCConvergenceParameters(max_bond_dimension=max_bond_dim, singval_mode='C')

    # Vertexes definition
    vertexes = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    # Plaquettes definition
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]
    # Number of link qubits
    num_links = shape[0]*(shape[1]-1) + shape[1]*(shape[0]-1)
    num_qubs = 2*len(vertexes) + num_links + 1

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
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes))]

    # ============= Initialize Hubbard circuit =============
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )
    qc = hbb.initialize_chessboard(qc, regs)
    original_qc = deepcopy(qc)
    for ii, pp in enumerate(plaquettes):
        qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )

    qc.barrier()
    evolution_instruction = hbb.adiabatic_operation(original_qc, regs, shape,
            hopping_constant, onsite_constant, chemical_potential,
            1, num_trotter_steps)
    qc = _preprocess_qk(qc, True, optimization=3)

    # ============= Apply Evolution =============
    u_and_d_exps = np.zeros((alpha_steps, 2*len(regs)) )
    ud_exps = np.zeros((alpha_steps, len(regs)) )
    entanglement_matter_links_exps = np.zeros(alpha_steps)
    idx = 0

    initial_state='Vacuum'

    qc1 = QuantumCircuit(*qc.qregs, *qc.cregs)
    qc1.append(evolution_instruction, range(qc.num_qubits))
    evolution_circ = _preprocess_qk(qc1, True, basis_gates=['u', 'cx', 'p', 'h', 'swap'], optimization=3)

    # ============= Prepare observables =============
    qc_ops = qtea.QCOperators()
    qc_ops.ops['z'] = np.array([[1, 0], [0, -1]])
    qc_ops.ops['y'] = np.array([[0, -1j], [1j, 0]])
    qc_ops.ops['x'] = np.array([[0, 1], [1, 0]])
    qc_obs = TNObservables()
    u_and_d_obs = obs.up_and_down_observable(original_qc, regs)
    for u_and_d in u_and_d_obs:
        qc_obs += u_and_d
    ud_obs = obs.updown_observable(original_qc, regs)
    for ud in ud_obs:
        qc_obs += ud
    qc_obs += TNObsBondEntropy()
    qc_obs += TNState2File('state.txt', 'F')

    # Add a first timestep with very strong U, the initial state setting
    alphas = np.linspace(0, 1, alpha_steps, endpoint=True)
    for alpha in tqdm(alphas):
        # Start from the state at the previous timestep
        if idx > 0:
            qc = deepcopy(evolution_circ)
            qc = qc.bind_parameters([alpha])

        # Simulate the circuit
        qcio = qtea.QCIO(inPATH='temp/in/', outPATH='temp/out/', initial_state=initial_state)
        res = run_simulation(qc,
                            convergence_parameters=conv_params,
                            io_info=qcio,
                            observables=qc_obs,
                            operators=qc_ops,
                            approach='PY',
                            transpilation_parameters=qk_transpilation_params(False)
                            )
        initial_state = MPS.from_tensor_list(res.mps, conv_params)

        for ii, site in enumerate(regs.values()):
            name = site.name
            ud_exps[idx, ii] = np.real(res.observables[name+'ud'])
        temp = []
        for mm in ('u', 'd'):
            for site in regs.values():
                name = site.name
                temp.append(np.real(res.observables[name+mm]))
        u_and_d_exps[idx, :] = temp
        entanglement_matter_links_exps[idx] = res.entanglement[(num_links-1, num_links)]*np.log2(np.e)

        idx += 1
        np.savetxt(os.path.join(dir_name, 'u_and_d.txt'), u_and_d_exps[:idx, :])
        np.savetxt(os.path.join(dir_name, 'ud.txt'), ud_exps[:idx, :])
        np.savetxt(os.path.join(dir_name, 'entanglement_matter_link.txt'), entanglement_matter_links_exps[:idx])

    # Save final state
    initial_state.write(os.path.join(dir_name, 'mps_state.txt'))

