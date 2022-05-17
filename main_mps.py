from hubbard.utils import lattice_str
from hubbard.mps.circuit import hubbard_circuit, initialize_chessboard
from hubbard.mps.stabilizers import apply_plaquette_stabilizers, apply_link_parity_stabilizer, apply_vertex_parity_stabilizer
from qcomps.py_emulator import QcMps
from qcomps import QCConvergenceParameters
import tn_py_frontend.observables as obs

shape = (4, 2)
regs, qc = hubbard_circuit(shape)
num_sites = qc.num_sites
qc = initialize_chessboard(qc, regs)

for ii in range(shape[0]-1):
    qc = apply_plaquette_stabilizers(qc, regs, f'plaquette{ii}', (ii,0) )


obsv = obs.TNObservables()
obsv += obs.TNObsProbabilities('E', prob_threshold=0.1)


# Apply error on site (0,0) rishon 'n'
#qc.x(regs['q(0, 0)']['n'] )
#qc.cx( [regs['q(0, 0)']['n'], regs['q(0, 1)']['s']] )

# Apply correction through link stabilizer
#qc = apply_link_parity_stabilizer(qc, regs, 'link1', (0, 1))

# Apply correction through vertex stabilizer
#qc = apply_vertex_parity_stabilizer(qc, regs, 'vertex1', (0, 0))

# Apply correction with plaquette
#qc = apply_plaquette_stabilizers(qc, regs, 'plaquette2', (0,0) )

qc.measure_observables('final', obsv)
simulator = QcMps(num_sites, convergence_parameters=QCConvergenceParameters(100))
res = simulator.run_from_qcirc(qc)
probs = res['final']['even_probability']['default']
print(qc.cregisters)

res = lattice_str(probs, regs, shape, use_1d_map=True )

print( res )

shapes = [ max(tens.shape) for tens in simulator]
print(shapes)
print(f'==== MAX BOND DIMENSION = {max(shapes)} ====')
with open(f'lattice_{shape[0]}x{shape[1]}.txt', 'w') as fh:
    fh.write(res)