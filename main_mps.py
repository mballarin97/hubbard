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
import pickle
from tqdm import tqdm
import os
import json
import time
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import numpy as np
import cupy as cp
from qiskit import AncillaRegister, ClassicalRegister
from qmatchatea import run_simulation
import qmatchatea as qtea
from qmatchatea.qk_utils import qk_transpilation_params
from qtealeaves.observables import TNObservables, TNObsBondEntropy, TNState2File, TNObsLocal
from qtealeaves.emulator import MPS

import hubbard as hbb
import hubbard.mps_observables as obs


if __name__ == '__main__':

    # ============= Initialize parameters of the simulation =============
    preproc_dir = "initial_states"
    with open(os.path.join(preproc_dir, "params.json"), 'rb') as fh:
        params = json.load(fh)

    # Shape of the lattice
    shape = params["shape"]
    # Hopping constant, usually called J or t
    hopping_constant = params["hopping_constant"]
    # Onsite constant, usually called U
    onsite_constant = params["onsite_constant"]
    # Number of timesteps
    num_timesteps = 500*10
    # Maximum bond dimension of the simulation
    max_bond_dim = 1024
    # Number of timesteps for a fixed alpha
    num_timesteps_before_meas = params["num_timesteps_before_measurement"]
    # Number of trotterization steps for the single timestep
    num_trotter_steps = num_timesteps_before_meas
    # dt for the evolution at each alpha
    dt = params["dt"]
    # If True, compute correlators
    compute_correlators = True
    state_idx = 24
    excitation = None#"spin"

    # Parameters dictionary for saving
    params['chi'] = max_bond_dim
    params['num_timesteps'] = num_timesteps
    #params.pop("num_timesteps_for_alpha")
    #params.pop("adiabatic_circ_generation_time")
    #params.pop("adiabatic_step_num_2qubit_gates")
    conv_params = qtea.QCConvergenceParameters(max_bond_dimension=max_bond_dim, singval_mode='C', cut_ratio=1e-8)
    backend = qtea.QCBackend(backend="PY", device="gpu")

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
        json.dump(params, fh, indent=4)

    # ============= Initialize qiskit variables =============
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes)+len(vertexes)+1)]
    with open(os.path.join(preproc_dir, "repulsive_evolution.pkl"), "rb") as fh:
        evolution_circ = pickle.load(fh)

    # ============= Initialize Hubbard circuit =============
    extra_leg = True if excitation == "charge" else False
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas, params["ordering"], extra_leg=extra_leg )
    tensor_list = qtea.read_mps(f"data/{state_idx}/mps_state.txt")

    if excitation == "charge":
        tensor_list = hbb.create_charge_excitation(qc, regs, tensor_list)
    elif excitation == "spin":
        tensor_list = hbb.create_spin_excitation(qc, regs, tensor_list)

    initial_state = MPS.from_tensor_list(tensor_list, conv_params)
    initial_state.iso_towards(0)
    qregs_names = [qreg.name for qreg in qc.qregs]

    # ============= Prepare observables =============
    z_on_qubits = np.zeros((num_timesteps, qc.num_qubits) )
    ud_exps = np.zeros((num_timesteps, len(regs)) )
    entanglement = np.zeros((num_timesteps, qc.num_qubits-1))
    timing = np.zeros(num_timesteps )
    # Initialize pauli matrices operators
    qc_ops = qtea.QCOperators()
    qc_ops.ops['z'] = cp.array([[1, 0], [0, -1]])
    qc_ops.ops['y'] = cp.array([[0, -1j], [1j, 0]])
    qc_ops.ops['x'] = cp.array([[0, 1], [1, 0]])
    # Initialize observables
    qc_obs = TNObservables()
    # Local observables, Z on each qubit
    qc_obs += TNObsLocal('z', 'z')
    # Double occupancy of the sites
    ud_names, uds = obs.updown_observable(qc, regs)
    for ud in uds:
        qc_obs += ud
    if compute_correlators:
        # Correlators
        corr_names, correlators = obs.correlators(qc, regs)
        for correlator in correlators:
            qc_obs += correlator
        correlators = np.zeros((num_timesteps, len(correlators)) )
    # Entanglement profile
    qc_obs += TNObsBondEntropy()
    # Save the state
    qc_obs += TNState2File('state.txt', 'F')

    # Prepare the alphas linearly spaced in 0,1 and then add
    # a series of 1s for the final evolution

    # ================= Main loop, over the alphas =================
    idx = 0
    singvals = []
    start = time.time()
    for timestep in range(num_timesteps):
        # ===== Inner loop, for each alpha evolve for 10 timesteps =====
        #if idx > 0:
        qc = deepcopy(evolution_circ)

        # Simulate the circuit
        qcio = qtea.QCIO(inPATH='temp/in/', outPATH='temp/out/', initial_state=initial_state)
        res = run_simulation(qc,
                            convergence_parameters=conv_params,
                            io_info=qcio,
                            observables=qc_obs,
                            backend = backend,
                            operators=qc_ops,
                            transpilation_parameters=qk_transpilation_params(False, tensor_compiler=False)
                            )
        initial_state = MPS.from_tensor_list(res.tens_net, conv_params)


        # Extract observables for each alpha (NOT EACH TIMESTEP)
        for ii, name in enumerate(ud_names):
            ud_exps[idx, ii] = np.real(res.observables[name])
        z_on_qubits[idx, :] = res.observables['z']
        #entanglement[idx, :] = np.array(list(res.entanglement.values()))*np.log2(np.e)
        if compute_correlators:
            for ii, name in enumerate(corr_names):
                correlators[idx, ii] = np.real(res.observables[name])
        singvals = np.hstack((singvals, np.sum(res.singular_values_cut) ))
        timing[idx] = time.time()-start

        idx += 1
        np.savetxt(os.path.join(dir_name, 'z_on_qubits.txt'), z_on_qubits[:idx, :],
                    header=' '.join(qregs_names) )
        np.savetxt(os.path.join(dir_name, 'ud.txt'), ud_exps[:idx, :],
                    header=' '.join(ud_names) )
        np.savetxt(os.path.join(dir_name, 'entanglement.txt'), entanglement[:idx, :])
        if compute_correlators:
            np.savetxt(os.path.join(dir_name, 'correlators.txt'), correlators[:idx, :],
                        header=' '.join(corr_names))
        np.savetxt( os.path.join(dir_name, 'singvals.txt'), singvals)
        np.savetxt( os.path.join(dir_name, 'time.txt'), timing)

    # Save final state
    initial_state.write(os.path.join(dir_name, 'mps_state.txt'))
