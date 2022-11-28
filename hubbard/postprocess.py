import json

import numpy as np
from qmatchatea import read_mps

__all__ = ["postprocess_data"]

def reorder_vector(params, extra_leg = False):
    """
    Reorder the output of a simulation such that
    you have first all the rishons and then the
    matter

    Parameters
    ----------
    params : dict
        parameters of the simulation
    extra_leg : bool, optional
        If you inserted a charge and so there is an extra leg,
        by default False

    Returns
    -------
    np.ndarray
        The array to sort the results of the z measures
    """
    num_rishons = (params["shape"][0]-1)*params["shape"][1] + params["shape"][0]*(params["shape"][1]-1) + int(extra_leg)
    new_ordering = []
    for oo in params["ordering"]:
        if oo < num_rishons:
            new_ordering.append(oo)
        else:
            new_ordering.append(2*oo-num_rishons )
            new_ordering.append(2*oo-num_rishons+1)
    return np.argsort(new_ordering)

def postprocess_data(idx, mps=False):
    """
    Postprocess the data

    Parameters
    ----------
    idx : int
        Index of the directory where the data are saved
    """
    # The result dir is fixed
    res_dir = f"data/{idx}/"

    with open(res_dir+"params.json") as fh:
        params = json.load(fh)

    # Local expectation value of z
    z_on_qubits = np.loadtxt(res_dir+"z_on_qubits.txt")
    # All the correlators
    correlators = np.loadtxt(res_dir+"correlators.txt")
    # Entanglement along the chain
    entanglement = np.loadtxt(res_dir+"entanglement.txt")
    # Cumulative singular values cut during the simulation
    singvals = np.loadtxt(res_dir+"singvals.txt")
    # Expectation value of ZZ on up and down
    ud = np.loadtxt(res_dir+"ud.txt")
    # Computational time used for the simulation
    true_times = np.loadtxt(res_dir+"time.txt")

    # MPS state, that might not be there
    if mps:
        try:
            mps_state = read_mps(res_dir+"mps_state.txt")
        except FileNotFoundError:
            mps_state = None
    else:
        mps_state = None

    results = {
        "params" : params,
        "entanglement" : entanglement,
        "singvals" : singvals,
        "true_times" : true_times,
        "mps_state" : mps_state
    }
    results.update( postprocess_ud(params, z_on_qubits, ud) )
    results.update( postprocess_correlators(params, results, correlators) )

    return results




def postprocess_ud(params, z_on_qubits, ud):
    """
    Postprocess the local Z measure and the two-qubit ZZ
    measures on up and down

    Parameters
    ----------
    params : dict
        parameters of the simulation
    z_on_qubits : np.ndarray
        expectation value of Z on all qubits
    ud : np.ndarray
        expectation value of ZZ on the ud qubits

    Returns
    -------
    dict
        dictionary of results
    """
    # Reorder the measured z in such a way that:
    # - First, all the vertical rishons are described
    # - Second, all the horizontal rishons are described
    # - Third, all the sites are described. In this case the outer loop is over the y coordinate
    extra_leg = True if params["excitation"] == "charge" else False
    num_rishons = (params["shape"][0]-1)*params["shape"][1] + params["shape"][0]*(params["shape"][1]-1) + int(extra_leg)
    num_sites = params["shape"][0]*params["shape"][1]

    reordered_z = z_on_qubits[:, reorder_vector(params, extra_leg)]
    # Map the measured Z into the occupancy number, n=(1-z)/2
    single_occupancy = (1-reordered_z)/2
    matter_occupancy = single_occupancy[:, num_rishons:]

    # Map the ZZ measurement into the double occupancy number, with the formula nn'=(1-z-z'+zz')/4
    matter_z = reordered_z[:, num_rishons:]
    even, odd = [ii for ii in range(0, 2*num_sites, 2)], [ii for ii in range(1, 2*num_sites, 2)]
    sum_matter_z = matter_z[:, even] + matter_z[:, odd]
    double_occupancy = (1- sum_matter_z + ud)/4

    # Compute charge density
    charge = matter_occupancy[:, even] + matter_occupancy[:, odd]
    # Compute spin density
    spin = 0.5* ( matter_occupancy[:, even] - matter_occupancy[:, odd] )
    spin[ :, [(ii+jj)%2==1 for jj in range(params["shape"][1]) for ii in range(params["shape"][0])] ] *= -1

    results = {
        "single_occupancy" : single_occupancy,
        "double_occupancy" : double_occupancy,
        "charge" : charge,
        "spin" : spin
    }

    return results

def postprocess_correlators(params, results, correlators):
    """
    Postprocess the correlators of spin and matter between
    all the possible combinations

    Parameters
    ----------
    params : dict
        parameters of the simulation
    results : dict
        dictionary of the new results
    correlators : np.ndarray
        expectation value of the correlators
    Returns
    -------
    dict
        dictionary of results
    """
    sites = [f'({ii}, {jj})' for jj in range(params["shape"][1]) for ii in range(params["shape"][0])]
    num_sites = params["shape"][0]*params["shape"][1]

    matter = ['u', 'd']
    ids = []
    for idx, site1 in enumerate(sites):
        for site2 in (sites[idx+1:]):
            for mm1 in (matter):
                for mm2 in (matter):
                    ids.append( site1+site2+mm1+mm2 )

    correlators_dict = {}
    for idx, id in enumerate(ids):
        correlators_dict[id] = correlators[:, idx]

    spin_correlator = np.zeros( (num_sites, num_sites, len(correlators)) )
    matter_correlator = np.zeros( (num_sites, num_sites, len(correlators)) )

    for idx, site1 in enumerate(sites):
        for jdx, site2 in enumerate(sites[idx+1:]):
            for mm1 in matter:
                for mm2 in matter:
                    id = site1+site2+mm1+mm2
                    sign = +1 if mm1==mm2 else -1
                    spin_correlator[idx, idx+jdx+1, :] += sign*correlators_dict[id]/16
                    matter_correlator[idx, idx+jdx+1, :] += correlators_dict[id]/16

            matter_correlator[idx, idx+jdx+1, :] -= results["charge"][:, idx]/8
            matter_correlator[idx, idx+jdx+1, :] -= results["charge"][:, idx+jdx]/8

    corr_results = {
        "matter_correlator" : matter_correlator,
        "spin_correlator" : spin_correlator
    }

    return corr_results