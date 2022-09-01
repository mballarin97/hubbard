# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import tn_py_frontend.observables as obs

def map_qiskit_idx(num_qubs, idx):
    return -(idx - num_qubs + 1)

def up_and_down_observable(qc, regs):
    """
    Return the observables to measure the expectation values of up and down
    matter in each site, which can later be connected to the charge and spin
    densities.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation

    Returns
    -------
    list
        list of observables. First, all the up expectations are
        returned, then all the down expectations. The order of measurement
        is the same you get by calling HubbardRegister.values()
    """
    # Total number of qubits
    num_qubs = qc.num_qubits

    matter = ['u', 'd']
    observables = []
    for site in regs.values():
        for mm in matter:
            # Prepare the operator, it just have Z on the up matter
            # of the site
            qubit = site[mm]
            qidx = [qc.find_bit(qubit).index]
            observables.append( obs.TNObsTensorProduct(site.name+mm, ['z'], [qidx]) )

    return observables

def updown_observable(qc, regs):
    """
    Return the observables to measure the expectation values of the
    PRODUCT up and down matter in each site.

    Parameters
    ----------
    qc : QuantumCircuit
        Quantum circuit describing the Hubbard model
    regs : HubbardRegister
        HubbardRegister of the simulation

    Returns
    -------
    list
        list of the expectation values. The order of measurement
        is the same you get by calling HubbardRegister.values()
    """
    # Total number of qubits
    num_qubs = qc.num_qubits

    observables = []
    matter = ['u', 'd']
    for site in regs.values():
        qidxs = []
        for mm in matter:
            # Prepare the operator, it just have Z on the matter
            # of the site
            qubit = site[mm]
            qidxs.append( [ qc.find_bit(qubit).index] )

        observables.append( obs.TNObsTensorProduct(site.name+'ud', ['z', 'z'], qidxs) )

    return observables
