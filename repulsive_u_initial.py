from cProfile import run
import hubbard as hbb
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator
import numpy as np
import matplotlib.pyplot as plt
from qmatchatea import run_simulation, QCConvergenceParameters
import tn_py_frontend.observables as obs

def old_build_circuit(qc, regs, qancilla, cancillas):
    # First two states (Off diagonal)
    qc.h(regs['q(0, 1)']['u'] )
    qc.x(regs['q(1, 0)']['d'] )
    qc.cx(regs['q(0, 1)']['u'], regs['q(0, 1)']['d'])
    qc.cx(regs['q(0, 1)']['u'], regs['q(1, 0)']['u'])
    qc.cx(regs['q(1, 0)']['u'], regs['q(1, 0)']['d'])
    qc.x(regs['q(0, 1)']['u'] )

    # Next diagonal
    qc.h(qancilla[0])
    qc.cswap(qancilla[0], regs['q(0, 1)']['u'], regs['q(0, 0)']['u'])
    qc.cswap(qancilla[0], regs['q(0, 1)']['d'], regs['q(0, 0)']['d'])
    qc.cswap(qancilla[0], regs['q(1, 0)']['u'], regs['q(1, 1)']['u'])
    qc.cswap(qancilla[0], regs['q(1, 0)']['d'], regs['q(1, 1)']['d'])
    qc.h(qancilla[0])
    qc.measure(qancilla[0], cancillas[0][0])
    qc.z(regs['q(0, 0)']['u']).c_if( cancillas[0], 1)
    qc.z(regs['q(0, 0)']['d']).c_if( cancillas[0], 1)
    qc.reset(qancilla[0])

    return qc

def build_circuit(qc, regs, qancilla, cancillas, shape):

    # Create upper row
    qc.h(regs[f'q(0, {shape[1]-1})']['u'])
    for ii in range(shape[0]):
        if (ii+shape[1]-1)%2 == 1:
            from_r, to_r = 'u', 'd'
        else:
            from_r, to_r = 'd', 'u'
        # Apply cx in the same site
        qc.cx(regs[f'q({ii}, {shape[1]-1})'][from_r],
                regs[f'q({ii}, {shape[1]-1})'][to_r])
        if ii == shape[0]-1:
            continue
        # Apply cx to next site
        qc.cx(regs[f'q({ii}, {shape[1]-1})'][to_r],
                regs[f'q({ii+1}, {shape[1]-1})'][to_r])

        if ii%2==0:
            qc.x(regs[f'q({ii}, {shape[1]-1})']['e'] )

    # flip the bits to have only one particle per site
    for ii in range(shape[0]):
        if (ii+shape[1]-1)%2 == 1:
            qc.x(regs[f'q({ii}, {shape[1]-1})']['u'] )
        else:
            qc.x(regs[f'q({ii}, {shape[1]-1})']['d'] )
    qc.barrier()

    # Complete with the other rows
    for jj in range(shape[1]-1, 0, -1):
        qc.h(qancilla[0])
        for ii in range(shape[0]):
            for mm in ('u', 'd'):
                qc.cswap(qancilla[0],
                            regs[f'q({ii}, {jj-1})'][mm],
                            regs[f'q({ii}, {jj})'][mm]
                        )
            if ii != shape[0]-1:
                qc.cswap(qancilla[0],
                            regs[f'q({ii}, {jj-1})']['e'],
                            regs[f'q({ii}, {jj})']['e']
                        )
        qc.h(qancilla[0])
        qc.measure(qancilla[0], cancillas[0][0])
        qc.z(regs[f'q({ii}, {jj})']['u']).c_if( cancillas[0], 1)
        qc.z(regs[f'q({ii}, {jj})']['d']).c_if( cancillas[0], 1)
        qc.reset(qancilla[0])
        qc.barrier()

    return qc

if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='double')
    shape = (4, 4)
    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, 'ca0'),ClassicalRegister(1, 'ca1')]
    vertexes = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    # Plaquettes definition
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]

    # Initialize Hubbard state
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )

    qc = build_circuit(qc, regs, qancilla, cancillas, shape)
    #print(qc)
    #qc.draw('mpl')
    #plt.show()
    #for ii, pp in enumerate(plaquettes):
    #    qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )
    for ii, vv in enumerate(vertexes):
        qc = hbb.apply_vertex_parity_stabilizer(qc, regs, qancilla, cancillas[0], vv)

    if False:
        res = execute(qc, backend=backend )
        results = res.result()
        statevect = results.get_statevector(qc).data
    else:
        obsv = obs.TNObservables()
        obsv += obs.TNObsProbabilities(num_samples=10000)
        #obsv += obs.TNState2File("state", "F")
        conv_params = QCConvergenceParameters(50)
        res = run_simulation(qc, convergence_parameters= conv_params, approach='PY', observables=obsv)
        statevect = res.measure_probabilities[0]
        #statevect = res.statevector

    state_str = hbb.lattice_str(qc, statevect, regs, shape)
    print(state_str)

    #print(len(statevect[np.abs(statevect)>1e-2]))
