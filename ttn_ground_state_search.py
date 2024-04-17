# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Ground state search using the Tree Tensor Network Ansatz as in the
first part of the paper.
Differently from the quantum circuit part, here we treat each matter
sites and rishons as a single super-site of local dimension 32.

Operators names:

C stands for corner
-> stands for px
^ stands for py
<- stands for mx

P stands for parity (penality on links)
Q{x} are the creation/destruction operators (used for hopping)
N_pair_half if the onsite term of the Hamiltonian

"""

from ed_lgt.operators import Z2_FermiHubbard_dressed_site_operators, Z2_FermiHubbard_gauge_invariant_states
import qtealeaves.modeling as mdl
import qtealeaves.observables as obs
from qtealeaves.convergence_parameters import TNConvergenceParameters
from qtealeaves.operators import TNOperators
from qtealeaves import QuantumGreenTeaSimulation
import numpy as np


if __name__=="__main__":
    dim = 2
    operators = Z2_FermiHubbard_dressed_site_operators(dim)
    basis, states = Z2_FermiHubbard_gauge_invariant_states(dim)

    projected_ops = {}
    for op in operators.keys():
        projected_ops[op] = basis["site"].transpose() @ operators[op] @ basis["site"]

    projected_ops["n_total"] = 0
    for s in "mp":
        for d in "xy":
            projected_ops["n_total"] += projected_ops[f"n_{s}{d}"]

    # Write down operators of the defermionized Hubbard
    operators = TNOperators()
    for key, val in projected_ops.items():
        operators.ops[key] = np.array(val.todense())
    operators.ops["id"] = np.identity(32)

    model = mdl.QuantumModel(dim, "L", name="Hubbard", map_type="HilbertCurveMap")#map_type="SnakeMap")
    # Tunneling
    for dir in ("x", "y"):
        for specie in ("up", "down"):
            model += mdl.TwoBodyTerm2D(
                operators=[f"Q{specie}_p{dir}_dag", f"Q{specie}_m{dir}"],
                shift=[1, 0] if dir == "x" else [0, 1],
                strength="t",
                prefactor=-1j,
                isotropy_xyz=False,
                add_complex_conjg=True
                )


    # Onsite interaction
    model += mdl.LocalTerm("N_pair_half", strength="U")

    # Link penalty
    model += mdl.LocalTerm("n_total", strength="P")
    for dir in ("x", "y"):
        model += mdl.TwoBodyTerm2D(
                operators=[f"n_p{dir}", f"n_m{dir}"],
                shift=[1, 0] if dir == "x" else [0, 1],
                prefactor=-2,
                strength="P",
                isotropy_xyz=False,
                )

    # Plaquette penalty
    model += mdl.PlaquetteTerm2D(
        operators=["C_px,py", "C_my,px", "C_py,mx", "C_mx,my"],
        strength="P",
        prefactor=-1
    )

    observables = obs.TNObservables()
    observables += obs.TNObsLocal("<S2>", "S2_psi")
    observables += obs.TNObsLocal("<N_tot>", "N_tot")
    observables += obs.TNObsLocal("<N_pair>", "N_pair")
    observables += obs.TNObsLocal("<n_mx>", "n_mx")
    observables += obs.TNObsLocal("<n_px>", "n_px")

    params = {
        "L" : (4, 4),
        "t" : 1,
        "U" : 8,
        "P" : 100
    }
    conv_params = TNConvergenceParameters(
        max_iter=10,
        max_bond_dimension=256,
        trunc_method="N",
        svd_ctrl="A",
        statics_method=2,
        data_type="C",
        ini_bond_dimension=32
    )


    sim = QuantumGreenTeaSimulation(
        model=model,
        operators=operators,
        convergence=conv_params,
        observables=observables,
        folder_name_input="ttn_sim/input",
        folder_name_output="ttn_sim/output",
        tn_type=5,
        tensor_backend=2,
        store_checkpoints=False,
        verbosity=10
    )

    # Run the simulation
    sim.run(params, delete_existing_folder=True)
