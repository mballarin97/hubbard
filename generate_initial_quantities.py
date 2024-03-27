import os
import pickle
import json
import time
import hubbard as hbb
import numpy as np
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit
from qmatchatea import run_simulation, QCConvergenceParameters, write_mps
from qmatchatea.qk_utils import qk_transpilation_params
from qmatchatea.preprocessing import _preprocess_qk
import qtealeaves.observables as obs

dir = "initial_states"
shape = (4, 2)
hopping_constant = 0.1
onsite_constant = 1
dt = 0.01
num_timesteps_for_alpha = 10
num_timesteps_before_measurement = 10
extra_leg = False

# 4x4 ordering
if shape == (4, 4):
    ordering = [24, 12, 25, 1, 13, 26, 14, 27, 3, 31, 17, 2, 30, 16, 5, 29,
                15, 0, 28, 4, 32, 18, 9, 33, 19, 6, 34, 20, 7, 35, 11, 39,
                23, 10, 38, 22, 37, 21, 8, 36]
    if extra_leg:
        # 4x2 ordering with extra rishon
        ordering = [oo+1 for oo in ordering]
        ordering = [0] + ordering
elif shape == (4, 2):
    # 4x2 ordering
    ordering = [ 14, 0, 10, 4, 11, 1, 7, 15, 8, 16, 2, 5, 12, 6, 13, 3, 9, 17

    ]

    if extra_leg:
        # 4x2 ordering with extra rishon
        ordering = [oo+1 for oo in ordering]
        ordering = ordering[:3] + [0] + ordering[3:]


params = {
    'shape' : shape,
    'ordering' : ordering,
    'hopping_constant' : hopping_constant,
    'onsite_constant' : onsite_constant,
    'dt' : dt,
    'num_timesteps_for_alpha' : num_timesteps_for_alpha,
    'num_timesteps_before_measurement' : num_timesteps_before_measurement
}

linear_params = qk_transpilation_params(True, basis_gates=['u', 'cx', 'p', 'h', 'swap'],
                        optimization=3, tensor_compiler=True)
generic_params = qk_transpilation_params(False, basis_gates=['u', 'cx', 'p', 'h', 'swap'],
                        optimization=3, tensor_compiler=False)
if __name__ == '__main__':

    # =================== Initialize variables ===================
    # Plaquettes and vertexes definition
    plaquettes = [(ii, jj) for jj in range(shape[1]-1) for ii in range(shape[0]-1) ]
    vertexes = [(ii, jj) for jj in range(shape[1]) for ii in range(shape[0])]
    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes)+len(vertexes)+1)]

    # =================== Initialize circuit ===================
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas, ordering=ordering, extra_leg=extra_leg )
    #qc = hbb.initialize_repulsive_rows(qc, regs, qancilla, cancillas[-1], shape, filling="h")
    qc = hbb.initialize_repulsive_chessboard(qc, regs)

    for ii, pp in enumerate(plaquettes):
        qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )
    for ii, vv in enumerate(vertexes):
        qc = hbb.apply_vertex_parity_stabilizer(qc, regs, qancilla, cancillas[len(plaquettes)+ii], vv)

    # =================== Run MPS simulation ===================
    obsv = obs.TNObservables()
    obsv += obs.TNObsProbabilities(num_samples=10000)
    obsv += obs.TNState2File("state", "F")
    conv_params = QCConvergenceParameters(500)
    start = time.time()
    res = run_simulation(qc, convergence_parameters= conv_params, observables=obsv)
    params['mps_simulation_time'] = time.time()-start
    statevect = res.measure_probabilities[0]
    max_bond_dim = np.array([ tens.shape for tens in res.tens_net]).flatten().max()
    params['initial_max_bond_dim'] = int(max_bond_dim)
    params['num_nonzero_states'] = len(statevect)

    # =================== Save MPS state ===================
    write_mps(os.path.join(dir, "initial_repulsive.txt"), res.tens_net)
#    with open(os.path.join(dir, "initial_repulsive_str.txt"), 'w' ) as fh:
#        state_str = hbb.lattice_str(qc, statevect, regs, shape)
#        fh.write(state_str)

    # =================== Generate adiabatic evolution circuit ===================
    if False:
        start = time.time()
        adiabatic_instruction = hbb.adiabatic_operation(qc, regs, shape, #superposition_adiabatic_operation
                    hopping_constant, onsite_constant, onsite_constant,
                    dt*num_timesteps_for_alpha, num_timesteps_for_alpha)
        qc1 = QuantumCircuit(*qc.qregs, *qc.cregs)
        qc1.append(adiabatic_instruction, range(qc.num_qubits))
        adiabatic_circ = _preprocess_qk(qc1, generic_params)
        params['adiabatic_step_num_2qubit_gates'] = int(adiabatic_circ.num_nonlocal_gates())
        adiabatic_circ = _preprocess_qk(qc1, linear_params)
        params['adiabatic_circ_generation_time'] = time.time()-start
        params['adiabatic_step_num_2qubit_gates_1d'] = int(adiabatic_circ.num_nonlocal_gates())

        # =================== Save adiabatic evolution circuit ===================
        with open(os.path.join(dir, "repulsive_adiabatic.pkl"), "wb") as fh:
            pickle.dump(adiabatic_circ, fh)


    # =================== Generate evolution circuit ===================
    start = time.time()
    evolution_instruction = hbb.evolution_operation(qc, regs, shape,
                hopping_constant, onsite_constant,
                dt*num_timesteps_before_measurement, num_timesteps_before_measurement)
    qc1 = QuantumCircuit(*qc.qregs, *qc.cregs)
    qc1.append(evolution_instruction, range(qc.num_qubits))
    evolution_circ = _preprocess_qk(qc1, generic_params)
    params['evolution_step_num_2qubit_gates'] = int(evolution_circ.num_nonlocal_gates())
    evolution_circ = _preprocess_qk(qc1, linear_params)
    params['evolution_circ_generation_time'] = time.time()-start
    params['evolution_step_num_2qubit_gates_1d'] = int(evolution_circ.num_nonlocal_gates())

    # =================== Save evolution circuit ===================
    with open(os.path.join(dir, "repulsive_evolution.pkl"), "wb") as fh:
        pickle.dump(evolution_circ, fh)
    print(params)

    # =================== Save important parameters ===================
    with open(os.path.join(dir, "params.json"), 'w') as fh:
        json.dump(params, fh, indent=4)
