import os
import pickle
import json
import time
import hubbard as hbb
import numpy as np
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit
from qmatchatea import run_simulation, QCConvergenceParameters, write_mps
from qmatchatea.preprocessing import _preprocess_qk
import tn_py_frontend.observables as obs

dir = "initial_states"
shape = (4, 4)
hopping_constant = 0.1
onsite_constant = 1
dt = 0.01
num_timesteps_for_alpha = 100
num_timesteps_before_measurement = 10
ordering = [24, 12, 25, 1, 13, 26, 14, 27, 3, 31, 17, 2, 30, 16, 5, 29,
            15, 0, 28, 4, 32, 18, 9, 33, 19, 6, 34, 20, 7, 35, 11, 39,
            23, 10, 38, 22, 37, 21, 8, 36]

params = {
    'shape' : shape,
    'ordering' : ordering,
    'hopping_constant' : hopping_constant,
    'onsite_constant' : onsite_constant,
    'dt' : dt,
    'num_timesteps_for_alpha' : num_timesteps_for_alpha,
    'num_timesteps_before_measurement' : num_timesteps_before_measurement
}

if __name__ == '__main__':

    # =================== Initialize variables ===================
    # Plaquettes and vertexes definition
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]
    vertexes = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes)+len(vertexes)+1)]

    # =================== Initialize circuit ===================
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas, ordering=ordering )
    qc = hbb.initialize_repulsive_rows(qc, regs, qancilla, cancillas[-1], shape)
    for ii, pp in enumerate(plaquettes):
        qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )
    for ii, vv in enumerate(vertexes):
        qc = hbb.apply_vertex_parity_stabilizer(qc, regs, qancilla, cancillas[len(plaquettes)+ii], vv)

    # =================== Run MPS simulation ===================
    obsv = obs.TNObservables()
    obsv += obs.TNObsProbabilities(num_samples=10000)
    obsv += obs.TNState2File("state", "F")
    conv_params = QCConvergenceParameters(50)
    start = time.time()
    res = run_simulation(qc, convergence_parameters= conv_params, approach='PY', observables=obsv)
    params['mps_simulation_time'] = time.time()-start
    statevect = res.measure_probabilities[0]
    max_bond_dim = np.array([ tens.shape for tens in res.mps]).flatten().max()
    params['initial_max_bond_dim'] = int(max_bond_dim)
    params['num_nonzero_states'] = len(statevect)

    # =================== Save MPS state ===================
    write_mps(os.path.join(dir, "initial_repulsive.txt"), res.mps)
    with open(os.path.join(dir, "initial_repulsive_str.txt"), 'w' ) as fh:
        state_str = hbb.lattice_str(qc, statevect, regs, shape)
        fh.write(state_str)


    # =================== Generate adiabatic evolution circuit ===================
    start = time.time()
    adiabatic_instruction = hbb.superposition_adiabatic_operation(qc, regs, shape,
                hopping_constant, onsite_constant,
                dt*num_timesteps_for_alpha, num_timesteps_for_alpha)
    qc1 = QuantumCircuit(*qc.qregs, *qc.cregs)
    qc1.append(adiabatic_instruction, range(qc.num_qubits))
    adiabatic_circ = _preprocess_qk(qc1, True, basis_gates=['u', 'cx', 'p', 'h', 'swap'], optimization=3)
    params['adiabatic_circ_generation_time'] = time.time()-start
    params['adiabatic_step_num_2qubit_gates'] = int(adiabatic_circ.num_nonlocal_gates())

    # =================== Save adiabatic evolution circuit ===================
    with open(os.path.join(dir, "repulsive_adiabatic.pkl"), "wb") as fh:
        pickle.dump(adiabatic_circ, fh)


    # =================== Generate adiabatic evolution circuit ===================
    start = time.time()
    evolution_instruction = hbb.evolution_operation(qc, regs, shape,
                hopping_constant, onsite_constant,
                dt*num_timesteps_before_measurement, num_timesteps_before_measurement)
    qc1 = QuantumCircuit(*qc.qregs, *qc.cregs)
    qc1.append(adiabatic_instruction, range(qc.num_qubits))
    evolution_circ = _preprocess_qk(qc1, True, basis_gates=['u', 'cx', 'p', 'h', 'swap'], optimization=3)
    params['evolution_circ_generation_time'] = time.time()-start
    params['evolution_step_num_2qubit_gates'] = int(adiabatic_circ.num_nonlocal_gates())

    # =================== Save evolution circuit ===================
    with open(os.path.join(dir, "repulsive_evolution.pkl"), "wb") as fh:
        pickle.dump(adiabatic_circ, fh)


    # =================== Save important parameters ===================
    with open(os.path.join(dir, "params.json"), 'w') as fh:
        json.dump(params, fh, indent=4)
