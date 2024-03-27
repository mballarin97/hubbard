# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import qtealeaves.observables as obs

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
    observables = []
    matter = ['u', 'd']
    names = []
    for site in regs.values():
        qidxs = []
        for mm in matter:
            # Prepare the operator, it just have Z on the matter
            # of the site
            qubit = site[mm]
            qidxs.append( [ qc.find_bit(qubit).index] )
        name = site.name+'ud'
        names.append(name)
        observables.append( obs.TNObsTensorProduct(name, ['z', 'z'], qidxs) )

    return names, observables

def correlators(qc, regs):
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
    observables = []
    matter = ['u', 'd']
    regs_vals = list(regs.values())
    names = []
    for ii, site_1 in enumerate(regs_vals):
        for site_2 in regs_vals[ii+1:]:
            for mm1 in matter:
                for mm2 in matter:

                    # Prepare the operator, it just have Z on the matter
                    # of the site
                    qubit1 = site_1[mm1]
                    qubit2 = site_2[mm2]
                    qidxs = [[ qc.find_bit(qubit1).index], [qc.find_bit(qubit2).index] ]

                    name = site_1.name+site_2.name+mm1+mm2
                    observables.append(
                        obs.TNObsTensorProduct(name, ['z', 'z'], qidxs)
                        )
                    names.append(name)

    return names, observables
