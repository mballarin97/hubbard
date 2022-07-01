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

    return parser

def plot_kinetic_term(kinetic_exp, save=False, path=''):
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
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(kinetic_exp, 'o--', color='navy')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(r'Kinetic term $\sum_{\langle ij\rangle}\langle \psi |Re( c^\dagger_{ij} c_{ij})|\psi\rangle $', fontsize=14)
    #ax.set_yscale('log')

    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, 'kinetic.pdf'), format='pdf')
    plt.show()

def plot_u_and_d_term(u_and_d, shape, save=False, path=''):
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
    """
    u_and_d = (-u_and_d+1)/2
    total_sites = np.prod(shape)
    rho_charge = u_and_d[:, :total_sites] + u_and_d[:, total_sites:]
    rho_spin = u_and_d[:, :total_sites] - u_and_d[:, total_sites:]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    minmin = np.min([np.min(rho_charge), np.min(rho_spin)])
    maxmax = np.max([np.max(rho_charge), np.max(rho_spin)])

    ax[0].set_title(r'Charge density $\rho_{c}=\langle n_{\uparrow}\rangle +\langle n_{\downarrow}\rangle$')
    im = ax[0].imshow(rho_charge, vmin=minmin, vmax=maxmax, aspect='auto')
    ax[0].set_ylabel('Time')

    ax[1].set_title(r'Spin density $\rho_{s}=\langle n_{\uparrow}\rangle -\langle n_{\downarrow}\rangle$')
    im2 = ax[1].imshow(rho_spin, vmin=minmin, vmax=maxmax, aspect='auto')


    for ii in range(2):
        ax[ii].set_xlabel('Site')
        ax[ii].set_xticks(np.arange(0, 4))
        ax[ii].set_xticklabels([f'({ii}, {jj})' for ii in range(shape[0]) for jj in range(shape[1])])

    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if save:
        plt.savefig(os.path.join(path, 'densities.pdf'), format='pdf')
    plt.show()

def plot_entanglement(entanglement, save=False, path=''):
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
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(entanglement, 'o--', color='forestgreen')
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel(r'Von Neumann entropy cutting in half the system $S_V$', fontsize=14)
    #ax.set_yscale('log')

    plt.grid()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, 'entanglement.pdf'), format='pdf')
    plt.show()