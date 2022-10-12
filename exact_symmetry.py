from copy import deepcopy
import hubbard as hbb
import numpy as np
import scipy.sparse as sp
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit
import qmatchatea as qtea
from qmatchatea.py_emulator import QcMps
from qmatchatea.preprocessing import _preprocess_qk
from tqdm import tqdm

from hubbard.hamiltonian_terms import hopping_hamiltonian
import os

shape = (4, 2)
# Interaction onsite constant
onsite_const = 1
# Hopping constant
hopping_const = 0.1

num_up = np.prod(shape)//4
num_down = np.prod(shape)//4
max_bond_dim = 10
conv_params = qtea.QCConvergenceParameters(max_bond_dimension=max_bond_dim, singval_mode='C')
all_states = hbb.all_possible_matter_states(shape, num_up, num_down)
print(f"There will be a total of {all_states.shape[0]} states")

# Hamiltonian of the symmetry sector
#symmetric_hamiltonian = np.zeros([all_states.shape[0]]*2, dtype=complex)
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


# ============= Initialize Hubbard circuit =============
regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )

for ii, vv in enumerate(vertexes):
    qc = hbb.apply_vertex_parity_stabilizer(qc, regs, qancilla, cancillas[len(plaquettes)+ii], vv)
for ii, pp in enumerate(plaquettes):
    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )

lin_qc = _preprocess_qk(qc)

# Generate all the correct MPS states that respect
# all the stabilizer checks
#all_states = all_states[[6, 12]]
mps_states = []
for state in all_states:
    temp_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    temp_state = QcMps(num_qubs, conv_params)
    for idx, qub_state in enumerate(state):
        if qub_state == 1:
            site = sites[ idx//2 ]
            specie = "u" if idx%2 == 0 else "d"
            temp_qc.x( regs[site][specie] )
    
    temp_state.run_from_qk(temp_qc)
    temp_state.run_from_qk(lin_qc)
    mps_states.append(temp_state)

    #print( hbb.lattice_str(qc, temp_state.to_statevector(), regs, shape) )
    #print("------")

# Define the Hamiltonian
total_hamiltonian = {}
onsite_hamiltonian = hbb.onsite_hamiltonian(qc, regs, shape, onsite_const)
hopping_hamiltonian = hbb.hopping_hamiltonian(qc, regs, shape, hopping_const)
total_hamiltonian.update(onsite_hamiltonian)
total_hamiltonian.update(hopping_hamiltonian)

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
        #print( hbb.lattice_str(qc, new_state.to_statevector(), regs, shape) )
        #print("-----------------")
        jdx = hbb.new_state_index(all_states, shape, idx, paulis, site_ordering)
        if jdx is not None:
            symmetric_hamiltonian[idx, jdx] += coeff*mps_states[jdx].contract(new_state)
        ## DEAD CODE, WAS CHECKING EACH POSSIBLE OVERLAP
        #happened = False
        #for jdx, statej in enumerate(mps_states):
        #    overlap = statej.contract(new_state)
        #    if np.abs(overlap) > 1e-9:
        #        print(f"State {idx} is connected with {jdx} through {paulis} with overlap {overlap}")
        #        print('---------')
        #        symmetric_hamiltonian[idx, jdx] += overlap*coeff
        #        happened = True
        #        break
        #if not happened:
        #    print("NOOO CONNECTION")

symmetric_hamiltonian = sp.csr_matrix(symmetric_hamiltonian)
#eigenvalues, eigenvectors = sp.linalg.eigsh(symmetric_hamiltonian, k=all_states.shape[0]-5 )
eigenvalues, eigenvectors = np.linalg.eigh(symmetric_hamiltonian.todense() )
#print(np.isclose( symmetric_hamiltonian-np.conj(symmetric_hamiltonian).T, np.zeros_like(symmetric_hamiltonian)).all() )
#print(symmetric_hamiltonian)
ordering = np.argsort(eigenvalues)
eigenvectors = eigenvectors[:, ordering]
eigenvalues = eigenvalues[ ordering]

save_dir = f"data/exact/{shape[0]}x{shape[1]}"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

sp.save_npz(os.path.join(save_dir, "symmetric_hamiltonian.npz"), symmetric_hamiltonian )
np.savetxt(os.path.join(save_dir, "eigenvalues.txt"), eigenvalues)
np.savetxt(os.path.join(save_dir, "eigenvectors.txt"), eigenvectors)
