import os
import matplotlib.pyplot as plt
from qplotting import Qplotter
from qplotting.utils import set_size_pt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


import hubbard as hbb

save_path = "images/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
results_1 = hbb.postprocess_data(19)
results_2 = hbb.postprocess_data(21)

svd = np.concatenate( (results_1["singvals"], results_2["singvals"]) )
print(f"The fidelity is {np.prod(1-svd)}")

spin = np.vstack( (results_1["double_occupancy"] - results_1["charge"], results_2["double_occupancy"] - results_1["charge"]) )
charge = np.vstack( (results_1["charge"], results_2["charge"]) )
spin = spin[:len(spin)//2]
charge = charge[:len(charge)//2]

timestep_bm = results_1["params"]["num_timesteps_before_measurement"]
dt = results_1["params"]["dt"]
total_timesteps = results_1["params"]["num_timesteps"] + results_2["params"]["num_timesteps"]
times = np.arange(0, total_timesteps)*timestep_bm*dt
times = times[:len(spin)]

###########################################################################
########################### SPIN-CHARGE VALUE #############################
###########################################################################
distance_charge = np.vstack( (
    charge[:, 0],
    charge[:, 1], #np.mean( np.hstack( (charge[:, 1].reshape(-1, 1), charge[:, 4].reshape(-1, 1)) ), axis=1),
    #charge[:, 5],
    charge[:, 2],
    #charge[:, 6],
    charge[:, 3],
    #charge[:, 7],
    ))
distance_spin = np.vstack( (
    spin[:, 0],
    spin[:, 1], #np.mean( np.hstack( (spin[:, 1].reshape(-1, 1), spin[:, 4].reshape(-1, 1)) ), axis=1),
    #spin[:, 5],
    spin[:, 2],
    #spin[:, 6],
    spin[:, 3],
    #spin[:, 7],
    ))

figname = os.path.join(save_path, ("spin_excitation_y=0.pdf") )
qplotter = Qplotter()
qplotter(nrows=2, ncols=1, figsize=set_size_pt(234, subplots=(2, 1)), sharex=True)
with qplotter as qplt:
    im0 = qplt.ax[0].imshow(distance_charge - distance_charge[:, 0].reshape(-1, 1), aspect='auto',
        interpolation='none', cmap="seismic", norm=TwoSlopeNorm(0))
    cbar = plt.colorbar(im0, ax=qplt.ax[0])
    cbar.ax.set_ylabel('Charge $N_i(t)-N_i(0)$', rotation=270, labelpad=15)

    qplt.ax[0].tick_params(axis='y', which='minor', left=False, right=False)
    qplt.ax[0].set_yticks( [ii for ii in range(4)])
    #qplt.ax[0].set_yticklabels( [0, 1, "$\sqrt{2}$", 2, "$\sqrt{5}$", 3, "$\sqrt{17}$"] )
    qplt.ax[0].set_yticklabels( [0, 1, 2, 3] )
    qplt.ax[0].set_ylabel("Distance from site (0,0)")

    im1 = qplt.ax[1].imshow(distance_spin - distance_spin[:, 0].reshape(-1, 1), aspect='auto',
        interpolation='none', cmap="seismic", norm=TwoSlopeNorm(0))
    cbar = plt.colorbar(im1, ax=qplt.ax[1])
    cbar.ax.set_ylabel('Spin $S^2_i(t)-S^2_i(0)$', rotation=270, labelpad=15)

    qplt.ax[1].tick_params(axis='y', which='minor', left=False, right=False)
    qplt.ax[1].set_yticks( [ii for ii in range(4)])
    #qplt.ax[1].set_yticklabels( [0, 1, "$\sqrt{2}$", 2, "$\sqrt{5}$", 3, "$\sqrt{17}$"] )
    qplt.ax[1].set_yticklabels( [0, 1, 2, 3] )
    qplt.ax[1].set_xticks( np.linspace(0, len(times)-1, 6, dtype=int) )
    qplt.ax[1].set_xticklabels( times[np.linspace(0, len(times)-1, 6, dtype=int) ].astype(int) )
    qplt.ax[1].set_xlabel("Time $t/U$")
    qplt.ax[1].set_ylabel("Distance from site (0,0)")

    qplt.savefig(figname)

###########################################################################
########################### SPIN-CHARGE CORRELATORS #######################
###########################################################################
shape = ( results_1["params"]["shape"][0], results_1["params"]["shape"][1])
sites = [f'({ii}, {jj})' for jj in range(shape[1]) for ii in range(shape[0])]

distances = np.sort( np.unique([ np.sqrt(ii**2+jj**2) for ii in range(shape[0]) for jj in range(shape[1]) ]) )[1:]
spin_correlator =  np.concatenate( (results_1["spin_correlator"], results_2["spin_correlator"] ), axis=2 )
matter_correlator =  np.concatenate( (results_1["matter_correlator"], results_2["matter_correlator"] ), axis=2 )

site = (0, 0)
site_idx = 0

spin_corr_distance = {str(np.round(dd, 3)) : [] for dd in distances}
matter_corr_distance = {str(np.round(dd, 3)) : [] for dd in distances}

for jdx, site2 in enumerate(sites[site_idx+1:]):
    site2 = np.array([int(site2[1]), int(site2[4]) ])
    dd = np.sqrt( (site2 **2).sum() )
    spin_corr_distance[str(np.round(dd, 3))].append( spin_correlator[site_idx, jdx+site_idx+1] )
    matter_corr_distance[str(np.round(dd, 3))].append( matter_correlator[site_idx, jdx+site_idx+1] )

spin_corr_distance = np.array( [ np.mean(sp, axis=0) for sp in spin_corr_distance.values() ] )
matter_corr_distance = np.array( [ np.mean(sp, axis=0) for sp in matter_corr_distance.values() ] )


qplotter = Qplotter()
qplotter(nrows=2, ncols=1, figsize=set_size_pt(234, subplots=(2, 1)), sharex=True)

figname = os.path.join(save_path, ("spin_excitation_correlators.pdf") )
with qplotter as qplt:

    im0 = qplt.ax[0].imshow(matter_corr_distance, aspect='auto', interpolation='none', cmap="seismic")
    cbar = plt.colorbar(im0, ax=qplt.ax[0])
    cbar.ax.set_ylabel('Charge correlator\n$\\langle N_{(0,0)}(t)N_i(t)\\rangle$', rotation=270, labelpad=25)

    qplt.ax[0].tick_params(axis='y', which='minor', left=False, right=False)
    qplt.ax[0].set_yticks( [ii for ii in range(6)])
    qplt.ax[0].set_yticklabels( [ 1, "$\sqrt{2}$", 2, "$\sqrt{5}$", 3, "$\sqrt{17}$"] )
    qplt.ax[0].set_ylabel("Distance from site (0,0)")

    im1 = qplt.ax[1].imshow(spin_corr_distance, aspect='auto',
        interpolation='none', cmap="seismic")
    cbar = plt.colorbar(im0, ax=qplt.ax[1])
    cbar.ax.set_ylabel('Spin correlator\n$\\langle S^z_{(0,0)}(t)S^z_i(t)\\rangle$', rotation=270, labelpad=25)

    qplt.ax[1].tick_params(axis='y', which='minor', left=False, right=False)
    qplt.ax[1].set_yticks( [ii for ii in range(6)])
    qplt.ax[1].set_yticklabels( [1, "$\sqrt{2}$", 2, "$\sqrt{5}$", 3, "$\sqrt{17}$"] )
    qplt.ax[1].set_xticks( np.linspace(0, len(times)-1, 6, dtype=int) )
    qplt.ax[1].set_xticklabels( times[np.linspace(0, len(times)-1, 6, dtype=int) ].astype(int) )
    qplt.ax[1].set_xlabel("Time $t/U$")
    qplt.ax[1].set_ylabel("Distance from site (0,0)")

    qplt.savefig(figname)
