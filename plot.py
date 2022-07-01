# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import json
import numpy as np
from sympy import symmetric_poly
import hubbard.plotter as hplt

if __name__ == '__main__':
    parser = hplt.plotter_parser()

    # Read the parameters
    args = parser.parse_args()

    # Index of the simulation
    idx = args.plot_index
    # If True, save the plots
    save = args.save
    # PATH to the files
    dir_path = os.path.join( 'data', str(idx) )

    # PATH for saving the plots
    if args.path is None:
        save_path = dir_path
    else:
        save_path = args.path

    with open(os.path.join(dir_path,'params.json' ), 'r') as fh:
        sim_params = json.load(fh)

    # First, check that the simulation makes sense through the
    # symmetry checks
    symmetry_checks = np.loadtxt(os.path.join(dir_path, 'symmetry_check.txt' ))
    if 1 in symmetry_checks:
        raise RuntimeError(
            f"""
            ===============================================
            ==================== ERROR ====================
            ===============================================
            The plaquette symmetry is not conserved,
            so no plots have been generated.
            The simulation {idx} is worthless.
            """)

    # Load and plot kinetic term
    kinetic = np.loadtxt(os.path.join(dir_path, 'kinetic.txt' ))
    hplt.plot_kinetic_term(kinetic, save, save_path)

    # Load and plot charge and spin density
    up_and_down = np.loadtxt(os.path.join(dir_path, 'u_and_d.txt' ))
    hplt.plot_u_and_d_term(up_and_down, sim_params['shape'], save, save_path)

    # Load and plot entanglement
    entanglement = np.loadtxt(os.path.join(dir_path, 'entanglement.txt' ))
    hplt.plot_entanglement(entanglement, save, save_path)
