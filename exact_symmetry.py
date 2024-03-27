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
Check that all the symmetries are enforced using either MPS or
exact diagonalization
"""

from copy import deepcopy
import hubbard as hbb
import numpy as np
import scipy.sparse as sp
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit
import qmatchatea as qtea
from qmatchatea.py_emulator import QcMps
from qmatchatea.preprocessing import _preprocess_qk
from tqdm import tqdm

import os

shape = (4, 2)
# Interaction onsite constant
onsite_const = 1
# Hopping constant
hopping_const = 0.1

num_up = np.prod(shape)//2
num_down = np.prod(shape)//2
max_bond_dim = 100
conv_params = qtea.QCConvergenceParameters(max_bond_dimension=max_bond_dim, singval_mode='C')
all_states = hbb.all_possible_matter_states(shape, num_up, num_down)
print(f"There will be a total of {all_states.shape[0]} states")

# Hamiltonian of the symmetry sector
symmetric_hamiltonian = sp.lil_matrix(tuple([all_states.shape[0]]*2), dtype=complex)

# Vertexes definition
vertexes = [(ii, jj) for jj in range(shape[1]) for ii in range(shape[0])]
sites = [f"q({ii}, {jj})" for jj in range(shape[1]) for ii in range(shape[0])]
site_ordering = []
idx = 0
for ii, jj in vertexes:
    if (ii+jj)%2==0:
        site_ordering += [idx, idx+1]
    else:
        site_ordering += [idx+1, idx]
    idx += 2
# Plaquettes definition
plaquettes = [(ii, jj) for jj in range(shape[1]-1) for ii in range(shape[0]-1) ]
# Number of link qubits
num_links = shape[0]*(shape[1]-1) + shape[1]*(shape[0]-1)
num_qubs = 2*len(vertexes) + num_links + 1
qancilla = AncillaRegister(1, 'a0')
cancillas = [ClassicalRegister(1, f'ca{ii}') for ii in range(len(plaquettes)+len(vertexes)+1)]
mps = True

# ============= Initialize Hubbard circuit =============
regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )

for ii, vv in enumerate(vertexes):
    qc = hbb.apply_vertex_parity_stabilizer(qc, regs, qancilla, cancillas[len(plaquettes)+ii], vv)
for ii, pp in enumerate(plaquettes):
    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )

lin_qc = _preprocess_qk(qc)
mps_states = []
all_states_strings = []
aaa = 0
for state in all_states:
    temp_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    temp_state = QcMps(num_qubs, 1, conv_params)
    for idx, qub_state in enumerate(state):
        if qub_state == 1:
            site = sites[ idx//2 ]
            specie = "u" if idx%2 == 0 else "d"
            temp_qc.x( regs[site][specie] )
    temp_state.run_from_qk(temp_qc)
    temp_state.run_from_qk(lin_qc)
    mps_states.append(temp_state)
    all_states_strings.append(
        list(temp_state.meas_projective(1).keys())[0][1:]
    )
    aaa += 1


# Define the Hamiltonian
total_hamiltonian = {}
onsite_hamiltonian = hbb.onsite_hamiltonian(qc, regs, shape, onsite_const)
hopping_hamiltonian = hbb.hopping_hamiltonian(qc, regs, shape, hopping_const)
total_hamiltonian.update(onsite_hamiltonian)
total_hamiltonian.update(hopping_hamiltonian)

if mps:
    # Fill the hamiltonian entries
    for idx, statei in tqdm(enumerate(mps_states)):
        for paulis, coeff in total_hamiltonian.items():
            new_state = deepcopy(statei)
            temp_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
            for pidx, pauli in enumerate(paulis[::-1]):
                if pauli == "X":
                    temp_qc.x(pidx)
                elif pauli == "Y":
                    temp_qc.y(pidx)
                elif pauli == "Z":
                    temp_qc.z(pidx)
            _ = new_state.run_from_qk(temp_qc)
            jdx = hbb.new_state_index(all_states, shape, idx, paulis, site_ordering)
            if jdx is not None:
                overlap = deepcopy(mps_states[jdx]).contract(new_state)
                symmetric_hamiltonian[idx, jdx] += coeff*overlap
                if idx == 25 and jdx == 16:
                    str1 = hbb.lattice_str(qc, new_state.meas_even_probabilities(0.01, qiskit_convention=True), regs, shape)
                    str2 = hbb.lattice_str(qc, mps_states[jdx].meas_even_probabilities(0.01, qiskit_convention=True), regs, shape)
                    print(str1 == str2 )
                    mps_states[jdx].right_canonize(0)
                    new_state.right_canonize(0)
                    print(f"State {idx} is connected with {jdx} through {paulis} with overlap {overlap}\n")
else:
    # Fill the hamiltonian entries
    print(all_states_strings)
    for idx, statei in tqdm(enumerate(all_states_strings)):
        for paulis, coeff in total_hamiltonian.items():
            new_state = ""
            phase = 1
            for ss, pauli in zip(all_states_strings[idx], paulis):
                if pauli == "X":
                    new_state += str( np.abs( int(ss)-1 ) )
                elif pauli == "Y":
                    new_state += str( np.abs( int(ss)-1 ) )
                    phase *= -1j if int(ss)==0 else 1j
                elif pauli == "Z":
                    new_state += str(ss)
                    phase *= 1 if int(ss)==0 else -1
                else:
                    new_state += str(ss)
            if np.sum( np.array(list(new_state[:-4]), dtype=int) )!= num_up+num_down:
                continue
            print(paulis, all_states_strings[idx], new_state)
            jdx = None
            for ii, state in enumerate(all_states_strings):
                if state[:-4] == new_state[:-4]:
                    jdx = ii
                    print(jdx)
                    break
            if jdx is not None:
                symmetric_hamiltonian[idx, jdx] += coeff*phase
        exit()

symmetric_hamiltonian = sp.csr_matrix(symmetric_hamiltonian)
eigenvalues, eigenvectors = np.linalg.eigh(symmetric_hamiltonian.todense() )
ordering = np.argsort(eigenvalues)
eigenvectors = eigenvectors[:, ordering]
eigenvalues = eigenvalues[ ordering]


save_dir = f"{shape[0]}x{shape[1]}"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

np.savetxt(os.path.join(save_dir, "symmetric_hamiltonian.txt"), symmetric_hamiltonian.todense() )
np.savetxt(os.path.join(save_dir, "eigenvalues.txt"), eigenvalues)
np.savetxt(os.path.join(save_dir, "eigenvectors.txt"), eigenvectors)
