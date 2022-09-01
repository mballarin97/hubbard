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

    if args.no_plot:
        plot = False
    else:
        plot = True

    with open(os.path.join(dir_path,'params.json' ), 'r') as fh:
        sim_params = json.load(fh)

    print('============ SIMULATION PARAMETERS ============')
    for key, val in sim_params.items():
        if key != 'U':
            print(f'\t {key} : {val}')
    sim_params['Ustep'] = sim_params['Ustep'] == "True"

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
    #kinetic = np.loadtxt(os.path.join(dir_path, 'kinetic.txt' ))
    #hplt.plot_kinetic_term(kinetic, sim_params, save, save_path, plot)

    # Load and plot charge and spin density
    up_and_down = np.loadtxt(os.path.join(dir_path, 'u_and_d.txt' ))
    hplt.plot_u_and_d_term(up_and_down, sim_params, save, save_path, plot)

    # Load and plot entanglement of half tghe system
    #entanglement = np.loadtxt(os.path.join(dir_path, 'entanglement_half.txt' ))
    #hplt.plot_half_entanglement(entanglement, sim_params, save, save_path, plot)

    # Load and plot entanglement between links and matter
    entanglement = np.loadtxt(os.path.join(dir_path, 'entanglement_matter_link.txt' ))
    hplt.plot_matter_link_entanglement(entanglement, sim_params, save, save_path, plot)

    # Load and plot ud term
    ud_term = np.loadtxt(os.path.join(dir_path, 'ud.txt' ))
    hplt.plot_ud_term(ud_term, sim_params, save, save_path, plot)
