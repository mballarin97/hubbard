from copy import deepcopy
from qiskit import ClassicalRegister, QuantumRegister
from .operators import generate_hopping
from hubbard.qiskit.circuit import hubbard_circuit
import numpy as np
from qiskit.circuit import Parameter

def hopping_circuit(shape):
    theta = Parameter('θ')
    regs, hop_circ = hubbard_circuit(shape, QuantumRegister(1), [ClassicalRegister(1)] )
    empty_circ = deepcopy(hop_circ)
    hop_circ.name = 'Hopping circuit'
    #for x_link in range(shape[0]-1):
        #for y_link in range(shape[1]-1):
    x_link = 0
    y_link = 0
    for species in ('u'):#, 'd'):
        operator, involved_regs = generate_hopping(regs, (x_link, y_link), species)

        for op, coef in operator.items():
            op = op.split('⊗')
            idx = 0
            temp_circ = deepcopy(empty_circ)
            for op_reg, reg in zip(op, involved_regs):
                for jdx, pauli in enumerate(op_reg):
                    if pauli == 'X':
                        # Apply ah Hadamard in some way
                        temp_circ.h(reg.qregister[jdx] )
                    elif pauli == 'Y':
                        # Apply the Rx(-pi/4) in some way
                        temp_circ.rx(-np.pi/4, reg.qregister[jdx] )
                    elif pauli == '1':
                        # Apply S in some way
                        temp_circ.s(reg.qregister[jdx] )

            for op_reg, reg in zip(op, involved_regs):
                for jdx, pauli in enumerate(op_reg):
                    if jdx < len(reg)-1 and idx == 0:
                        temp_circ.cx(reg.qregister[jdx], reg.qregister[jdx+1])
                    elif idx == 0:
                        temp_circ.cx(reg.qregister[idx], involved_regs[1].qregister[0])
                    elif jdx < len(reg)-2:
                        temp_circ.cx(reg.qregister[jdx], reg.qregister[jdx+1])
                idx += 1
            herm_conj = temp_circ.reverse_ops()
            temp_circ.cp(theta, reg.qregister[jdx-1], reg.qregister[jdx])

            hop_circ = hop_circ.compose(temp_circ)
            hop_circ = hop_circ.compose(herm_conj)
            hop_circ.barrier()

    return hop_circ