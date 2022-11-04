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

def wigner_onsite(shape, onsite):
    hamiltonian = {}
    string = list("I"*2*np.prod(shape))
    for ii in range(0, 2*np.prod(shape), 2):
        temp = deepcopy(string)
        temp[ii:ii+2] = "ZZ"
        hamiltonian["".join(temp)] = onsite/4
    return hamiltonian

def generate_hopping(shape, link_idx, species):
    # If the y component of the index is even
    # the link is horizontal, otherwise vertical
    is_horizontal = (link_idx[1] == 'h')
    link_idx = int(link_idx[2:])
    string_hop = np.repeat("II", np.prod(shape)).reshape(shape)
    if is_horizontal:
        ypos = link_idx//(shape[0]-1)
        xpos = link_idx%(shape[0]-1)
        from_site = (xpos, ypos)
        to_site = (xpos+1, ypos)
    else:
        ypos = link_idx//shape[0]
        xpos = link_idx%shape[0]
        from_site = (xpos, ypos)
        to_site = (xpos, ypos+1)

    if species == "u":
        string_hop[from_site[0], from_site[1]] = "XZ"
        string_hop[to_site[0], to_site[1]] = "XI"
    else:
        string_hop[from_site[0], from_site[1]] = "IX"
        string_hop[to_site[0], to_site[1]] = "ZX"

    final_str = ""
    for jj in range(shape[1]):
        for ii in range(shape[0]):
            final_str += string_hop[ii, jj]
    index_first_x = final_str.find("X")
    index_last_x = final_str.rfind("X")

    final = list(final_str)
    final[index_first_x+1:index_last_x] = "Z"*(index_last_x-index_first_x-1)
    other = deepcopy(final)
    other[index_first_x] = "Y"
    other[index_last_x] = "Y"
    finals = [
        "".join(final),
        "".join(other)
    ]

    return finals

def wigner_hopping(shape, hopping):
    hamiltonian = {}
    # Links available in lattice of given shape
    vert_links = [f'lv{ii}' for ii in range(shape[0]*(shape[1]-1))]
    horiz_links = [f'lh{ii}' for ii in range(shape[1]*(shape[0]-1))]
    avail_links = vert_links + horiz_links

    for link_idx in avail_links:
        for species in ("u", "d"):
            operators = generate_hopping(shape, link_idx, species)
            for op in operators:
                hamiltonian[op] = hopping/2

    return hamiltonian


shape = (4, 2)
# Interaction onsite constant
onsite_const = 1
# Hopping constant
hopping_const = 0.1

num_up = np.prod(shape)//2
num_down = np.prod(shape)//2
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

total_hamiltonian = {}
onsite_hamiltonian = wigner_onsite(shape, onsite_const)
hopping_hamiltonian = wigner_hopping(shape, hopping_const)
total_hamiltonian.update(onsite_hamiltonian)
total_hamiltonian.update(hopping_hamiltonian)
#for ii in total_hamiltonian:
#    print(ii)

all_states_strings = np.array([ "".join(list(ss)) for ss in all_states.astype(str) ])

# Fill the hamiltonian entries
for idx, statei in tqdm(enumerate(all_states)):
    #print(all_states_strings[idx])
    for paulis, coeff in total_hamiltonian.items():
        new_state = ""
        phase = 1
        for ss, pauli in zip(statei, paulis):
            if pauli == "X":
                new_state += str( np.abs( ss-1 ) )
            elif pauli == "Y":
                new_state += str( np.abs( int(ss)-1 ) )
                phase *= -1j if ss==0 else 1j
            elif pauli == "Z":
                new_state += str(ss)
                phase *= 1 if ss==0 else -1
            else:
                new_state += str(ss)

        jdx = None
        for ii, state in enumerate(all_states_strings):
            if state == new_state:
                jdx = ii
                break
        if jdx is not None:
            #print(f"State {idx} is connected with {jdx} through {paulis} with overlap {phase}")
            symmetric_hamiltonian[idx, jdx] += coeff*phase

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

np.savetxt(os.path.join(save_dir, "symmetric_hamiltonian_jw.txt"), symmetric_hamiltonian.todense() )
np.savetxt(os.path.join(save_dir, "eigenvalues_jw.txt"), eigenvalues)
np.savetxt(os.path.join(save_dir, "eigenvectors_jw.txt"), eigenvectors)
