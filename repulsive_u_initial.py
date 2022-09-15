import hubbard as hbb
from qiskit import AncillaRegister, ClassicalRegister, execute
from qiskit.providers.aer import StatevectorSimulator
import numpy as np
import pickle


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

def build_circuit(qc, regs, qancilla, cancillas):
    # First two states (Upper part)
    qc.h(regs['q(0, 1)']['u'] )
    qc.x(regs['q(1, 1)']['d'] )
    qc.cx(regs['q(0, 1)']['u'], regs['q(0, 1)']['d'])
    qc.cx(regs['q(0, 1)']['u'], regs['q(1, 1)']['u'])
    qc.cx(regs['q(1, 1)']['u'], regs['q(1, 1)']['d'])
    qc.x(regs['q(0, 1)']['u'] )
    qc.x(regs['q(0, 1)']['e'] )

    # Next diagonal
    qc.h(qancilla[0])
    qc.cswap(qancilla[0], regs['q(0, 1)']['u'], regs['q(0, 0)']['u'])
    qc.cswap(qancilla[0], regs['q(0, 1)']['d'], regs['q(0, 0)']['d'])
    qc.cswap(qancilla[0], regs['q(1, 0)']['u'], regs['q(1, 1)']['u'])
    qc.cswap(qancilla[0], regs['q(1, 0)']['d'], regs['q(1, 1)']['d'])
    qc.cswap(qancilla[0], regs['q(1, 0)']['w'], regs['q(1, 1)']['w'])
    qc.h(qancilla[0])
    qc.measure(qancilla[0], cancillas[0][0])
    qc.z(regs['q(0, 0)']['u']).c_if( cancillas[0], 1)
    qc.z(regs['q(0, 0)']['d']).c_if( cancillas[0], 1)
    qc.reset(qancilla[0])

    return qc

if __name__ == '__main__':
    # Simulation backend
    backend = StatevectorSimulator(precision='double')
    shape = (2, 2)
    # Initialize ancilla bits
    qancilla = AncillaRegister(1, 'a0')
    cancillas = [ClassicalRegister(1, 'ca0'),ClassicalRegister(1, 'ca1')]
    vertexes = [(ii, jj) for ii in range(shape[0]) for jj in range(shape[1])]
    # Plaquettes definition
    plaquettes = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]-1) ]

    # Initialize Hubbard state
    shape = (2, 2)
    regs, qc = hbb.hubbard_circuit(shape, qancilla, cancillas )

    qc = build_circuit(qc, regs, qancilla, cancillas)

    for ii, pp in enumerate(plaquettes):
        qc = hbb.apply_plaquette_stabilizers(qc, regs, qancilla[0], cancillas[ii], pp )
    for ii, vv in enumerate(vertexes):
        qc = hbb.apply_vertex_parity_stabilizer(qc, regs, qancilla, cancillas[0], vv)

    qc.draw('mpl', filename="repulsive_state_preparation.pdf")
    res = execute(qc, backend=backend )
    results = res.result()
    statevect = results.get_statevector(qc).data
    state_str = hbb.lattice_str(qc, statevect, regs, shape)
    print(state_str)

    print(len(statevect[np.abs(statevect)>1e-2]))
