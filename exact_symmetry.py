from copy import deepcopy
import hubbard as hbb
import numpy as np
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit
import qmatchatea as qtea
from qmatchatea.py_emulator import QcMps

shape = (2, 2)
# Interaction onsite constant
onsite_const = 1
# Hopping constant
hopping_const = 0.1

num_up = 1
num_down = 1
max_bond_dim = 10
conv_params = qtea.QCConvergenceParameters(max_bond_dimension=max_bond_dim, singval_mode='C')
all_states = hbb.all_possible_matter_states(shape, num_up, num_down)
print(f"There will be a total of {all_states.shape[0]} states")

# Hamiltonian of the symmetry sector
symmetric_hamiltonian = np.zeros([all_states.shape[0]]*2, dtype=complex)

# Vertexes definition
vertexes = [(ii, jj) for jj in range(shape[1]) for ii in range(shape[0])]
sites = [f"q({ii}, {jj})" for jj in range(shape[1]) for ii in range(shape[0])]
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

# Generate all the correct MPS states that respect
# all the stabilizer checks
mps_states = []
for state in all_states[:]:
    temp_qc = QuantumCircuit(*qc.qregs, *qc.cregs)
    temp_state = QcMps(num_qubs, conv_params)
    for idx, qub_state in enumerate(state):
        if qub_state == 1:
            site = sites[ idx//2 ]
            specie = "u" if idx%2 == 0 else "d"
            temp_qc.x( regs[site][specie] )
    
    temp_state.run_from_qk(temp_qc)
    temp_state.run_from_qk(qc)
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
for idx, state in enumerate(mps_states):
    for paulis, coeff in total_hamiltonian.items():
        new_state = deepcopy(state)
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
        for jdx, state in enumerate(mps_states):
            overlap = state.contract(new_state)
            if np.abs(overlap) > 1e-9:
                #print(f"State {idx} is connected with {jdx} through {paulis} with overlap {overlap}")
                symmetric_hamiltonian[idx, jdx] += overlap*coeff
                break

np.savetxt("output.txt", symmetric_hamiltonian )


