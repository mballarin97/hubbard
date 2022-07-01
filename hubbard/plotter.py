# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import argparse

def plotter_parser():
    """
    Parser to read from command line the parameters for the plotting

    Returns
    -------
    argparse.ArgumentParser
        The parser object
    """
    parser = argparse.ArgumentParser(prog='HUBBARD_DIGITAL_EVOLUTION',
        description="""
        Plotter for the results
        """)
    parser.add_argument('plot_index', metavar='idx', type=int,
                    help='The integer identifier of the simulation to plot')
    parser.add_argument('--save', nargs='?', const=True, type=bool, default=False,
        help='If provided, save the results')
    parser.add_argument('--path', nargs='?', const=None, type=str, default=None,
        help='If provided, save in this PATH. Otherwise, save in data/idx/.')
    parser.add_argument('--no_plot', nargs='?', const=False, type=bool, default=True,
        help='If provided, do not plot the results')

    return parser

def plot_kinetic_term(kinetic_exp, save=False, path='', plot=True):
    """
    Plot the kinetic term

    Parameters
    ----------
    kinetic_exp : np.ndarray
        Kinetic term expectation value
    save : bool, optional
        If True, save the pdf file, by default False
    path ; str, optional
        PATH where to save the file
    plot : bool, optional
        If True, use plt.show(). Default to True.
    """
    _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(kinetic_exp, 'o--', color='navy')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(r'Kinetic term $\sum_{\langle ij\rangle}\langle \psi |Re( c^\dagger_{ij} c_{ij})|\psi\rangle $', fontsize=14)
    #ax.set_yscale('log')

    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, 'kinetic.pdf'), format='pdf')
    if plot:
        plt.show()

def plot_u_and_d_term(u_and_d, shape, save=False, path='', plot=True):
    """
    Plot the charge and spin densities

    Parameters
    ----------
    u_and_d : np.ndarray
        u_and_d term expectation value
    save : bool, optional
        If True, save the pdf file, by default False
    path ; str, optional
        PATH where to save the file
    plot : bool, optional
        If True, use plt.show(). Default to True.
    """
    u_and_d = (-u_and_d+1)/2
    total_sites = np.prod(shape)
    rho_charge = u_and_d[:, :total_sites] + u_and_d[:, total_sites:]
    rho_spin = u_and_d[:, :total_sites] - u_and_d[:, total_sites:]
    sites = [f'({ii}, {jj})' for ii in range(shape[0]) for jj in range(shape[1])]

    ###################################################
    # Plotting The full evolution with imshow

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    minmin = np.min([np.min(rho_charge), np.min(rho_spin)])
    maxmax = np.max([np.max(rho_charge), np.max(rho_spin)])

    ax[0].set_title(r'Charge density $\rho_{c}=\langle n_{\uparrow}\rangle +\langle n_{\downarrow}\rangle$')
    im = ax[0].imshow(rho_charge, vmin=minmin, vmax=maxmax, aspect='auto')
    ax[0].set_ylabel('Time')

    ax[1].set_title(r'Spin density $\rho_{s}=\langle n_{\uparrow}\rangle -\langle n_{\downarrow}\rangle$')
    _ = ax[1].imshow(rho_spin, vmin=minmin, vmax=maxmax, aspect='auto')


    for ii in range(2):
        ax[ii].set_xlabel('Site')
        ax[ii].set_xticks(np.arange(0, 4))
        ax[ii].set_xticklabels(sites)

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if save:
        plt.savefig(os.path.join(path, 'densities_colors.pdf'), format='pdf')
    if plot:
        plt.show()

    ###################################################
    # Plotting only the profile
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    colors = matplotlib.cm.get_cmap('Dark2')
    ax[0].set_ylabel(r'Charge density $\rho_{c}=\langle n_{\uparrow}\rangle +\langle n_{\downarrow}\rangle$')
    ax[0].set_xlabel('Time')

    markers = list(Line2D.markers.keys())
    for ii, rho in enumerate(rho_charge.T):
        ax[0].plot(rho, markers[ii]+'--', color=colors(ii), label=sites[ii], alpha=0.8)
    ax[0].legend()

    ax[1].set_ylabel(r'Spin density $\rho_{s}=\langle n_{\uparrow}\rangle -\langle n_{\downarrow}\rangle$')
    ax[1].set_xlabel('Time')

    for ii, rho in enumerate(rho_spin.T):
        ax[1].plot(rho, markers[ii]+'--', color=colors(ii), label=sites[ii], alpha=0.8)
    ax[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, 'densities_lines.pdf'), format='pdf')
    if plot:
        plt.show()

def plot_entanglement(entanglement, save=False, path='', plot=True):
    """
    Plot the entanglement term

    Parameters
    ----------
    kinetic_exp : np.ndarray
        Kinetic term expectation value
    save : bool, optional
        If True, save the pdf file, by default False
    path ; str, optional
        PATH where to save the file
    plot : bool, optional
        If True, use plt.show(). Default to True.
    """
    _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(entanglement, 'o--', color='forestgreen')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(r'Von Neumann entropy cutting in half the system $S_V$', fontsize=14)
    #ax.set_yscale('log')

    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, 'entanglement.pdf'), format='pdf')
    if plot:
        plt.show()

def plot_ud_term(ud_term, shape, save=False, path='', plot=True):
    """
    Plot the charge and spin densities

    Parameters
    ----------
    ud_term : np.ndarray
        ud term expectation value
    save : bool, optional
        If True, save the pdf file, by default False
    path ; str, optional
        PATH where to save the file
    plot : bool, optional
        If True, use plt.show(). Default to True.
    """
    sites = [f'({ii}, {jj})' for ii in range(shape[0]) for jj in range(shape[1])]

    _, ax = plt.subplots(figsize=(8, 6))

    colors = matplotlib.cm.get_cmap('Dark2')
    markers = list(Line2D.markers.keys())
    for ii, rho in enumerate(ud_term.T):
        ax.plot(rho, markers[ii]+'--', color=colors(ii), label=sites[ii], alpha=0.8)
    ax.legend()
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(r'Joint expectation \langle n_{\uparrow}n_{\downarrow}\rangle', fontsize=14)
    #ax.set_yscale('log')

    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, 'ud.pdf'), format='pdf')
    if plot:
        plt.show()
