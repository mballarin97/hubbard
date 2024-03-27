import os
import matplotlib.pyplot as plt
from qplotting import Qplotter
from qplotting.utils import set_size_pt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import argrelextrema

import hubbard as hbb


def plot_lattice(ax, shape, zz):
    for jj in range(shape[1]):
        ax.plot([0, shape[0]-1], [jj, jj], [zz, zz], color="black", alpha=0.7, lw=0.5)
    for ii in range(shape[0]):
        ax.plot([ii, ii], [0, shape[1]-1], [zz, zz], color="black", alpha=0.7, lw=0.5)

save_path = "images/"
if not os.path.isdir(save_path):
    os.mkdir(save_path)
results_1 = hbb.postprocess_data(19)
results_2 = hbb.postprocess_data(22)

svd =  results_1["singvals"]
print(f"The fidelity is {np.prod(1-svd)}")

spin = -2*results_1["double_occupancy"] + results_1["charge"]
charge = results_2["charge"]

timestep_bm = results_1["params"]["num_timesteps_before_measurement"]
dt = results_1["params"]["dt"]
total_timesteps = results_1["params"]["num_timesteps"]
times = np.arange(0, total_timesteps)*timestep_bm*dt
times = times[:len(spin)]

###########################################################################
########################### SPIN-CHARGE VALUE #############################
###########################################################################

figname = os.path.join(save_path, ("spin_charge_separation_snapshots.pdf") )
qplotter = Qplotter()
#qplotter(nrows=2, ncols=1, figsize=set_size_pt(234, subplots=(2, 1)), sharex=True)

cm = plt.cm.get_cmap('seismic')
coords = np.array([ [ii, jj] for jj in range(2) for ii in range(4)])

with qplotter as qplt:

    qplt.fig = plt.figure(figsize=set_size_pt(2*234, subplots=(2, 2)) )

    qplt.ax = qplt.add_subplot(2, 2, 1, projection='3d')
    qplt.tick_params(axis='x', which='minor', left=False, right=False)
    qplt.tick_params(axis='y', which='minor', top=False, bottom=False)
    qplt.tick_params(axis='z', which='minor', right=False)
    qplt.tick_params(axis='x', which='major', pad=-5)
    qplt.tick_params(axis='y', which='major', pad=-5)
    qplt.tick_params(axis='z', which='major', pad=-2)
    qplt.set_zlabel("Time $\\tau/U$", labelpad=-6, rotation=270)
    qplt.set_title("Charge excitation")
    #qplt.set_xticks([])
    qplt.set_yticks([0, 1])
    qplt.grid(False)
    qplt.ax.text2D(-0.1, 0.05, r"$\textbf{(a)}$")

    selected_times = [ 10, 20, 30 ]
    norm_charge = charge - charge[0, :].reshape(1, -1)
    cols = norm_charge[selected_times, :].reshape(-1)
    xvals = np.tile(coords[:, 0], len(selected_times) )
    yvals = np.tile(coords[:, 1], len(selected_times) )
    zvals = np.array( [np.repeat(ii, len(coords)) for ii in selected_times ] ).reshape(-1)

    qplt.set_zticks(selected_times)
    qplt.set_zticklabels(times[selected_times])

    for zz in selected_times:
        plot_lattice(qplt.ax, (4, 2), zz)
    sc = qplt.ax.scatter(xvals, yvals, zvals, c=cols, cmap = cm, norm=TwoSlopeNorm(0),
        edgecolors="black", alpha=1, linewidth=0.5)
    cbar = qplt.colorbar(sc, ax = qplt.ax, pad=0.2)
    cbar.ax.set_ylabel('Charge $N_j(\\tau)-N_j(0)$', rotation=90, labelpad=5)
    qplt.ax.view_init(elev=10)


    # SPIN
    qplt.ax = qplt.add_subplot(2, 2, 2, projection='3d')
    qplt.tick_params(axis='x', which='minor', left=False, right=False)
    qplt.tick_params(axis='y', which='minor', top=False, bottom=False)
    qplt.tick_params(axis='z', which='minor', right=False)
    qplt.tick_params(axis='x', which='major', pad=-5)
    qplt.tick_params(axis='y', which='major', pad=-5)
    qplt.tick_params(axis='z', which='major', pad=-2)
    qplt.set_zlabel("Time $\\tau/U$", labelpad=-6)
    qplt.set_title("Spin excitation")
    qplt.ax.text2D(-0.1, 0.05, r"$\textbf{(c)}$")
    #qplt.set_xticks([])
    qplt.set_yticks([0, 1])
    qplt.grid(False)

    selected_times = [ 10, 20, 30 ]
    norm_spin = spin - spin[0, :].reshape(1, -1)
    cols = norm_spin[selected_times, :].reshape(-1)
    xvals = np.tile(coords[:, 0], len(selected_times) )
    yvals = np.tile(coords[:, 1], len(selected_times) )
    zvals = np.array( [np.repeat(ii, len(coords)) for ii in selected_times ] ).reshape(-1)

    qplt.set_zticks(selected_times)
    qplt.set_zticklabels(times[selected_times])

    for zz in selected_times:
        plot_lattice(qplt.ax, (4, 2), zz)
    sc = qplt.ax.scatter(xvals, yvals, zvals, c=cols, cmap = cm, norm=TwoSlopeNorm(0),
        edgecolors="black", alpha=1, linewidth=0.5)
    cbar = qplt.colorbar(sc, ax = qplt.ax, pad=0.2)
    cbar.ax.set_ylabel('Spin $S^2_j(\\tau)-S^2_j(0)$', rotation=90, labelpad=5)
    qplt.ax.view_init(elev=10)


    qplt.ax = qplt.add_subplot(2, 2, 3)
    qplt.plot(times, norm_charge[:, 0], label="(0, 0)", color="blue")
    qplt.plot(times, norm_charge[:, 4], label="(1, 1)", color="green")
    #qplt.plot(times, norm_charge[:, 2], label="(2, 0)", color="yellow")
    qplt.plot(times, norm_charge[:, 3], label="(3, 0)", color="red")
    qplt.set_xlabel("Time $\\tau/U$")
    qplt.set_ylabel("Charge $N_j(\\tau)-N_j(0)$")
    qplt.legend(frameon=False, title="Site index")
    qplt.text(-5, 1.1, r"$\textbf{(b)}$")

    qplt.ax = qplt.add_subplot(2, 2, 4)
    qplt.plot(times, norm_spin[:, 0], label="(0, 0)", color="blue")
    qplt.plot(times, norm_spin[:, 4], label="(1, 1)", color="green")
    #qplt.plot(times, norm_spin[:, 2], label="(2, 0)", color="yellow")
    qplt.plot(times, norm_spin[:, 3], label="(3, 0)", color="red")
    qplt.set_xlabel("Time $\\tau/U$")
    qplt.set_ylabel("Spin $S^2_j(\\tau)-S^2_j(0)$")
    #qplt.yaxis.set_label_position("right")
    #qplt.yaxis.tick_right()
    qplt.text(-5, 0.0205, r"$\textbf{(d)}$")



    plt.subplots_adjust(wspace=0.5)
    qplt.savefig(figname)

