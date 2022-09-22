# This code is part of hubbard.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit import QuantumRegister
from copy import deepcopy
import numpy as np

EVEN_SITE_LIST = ['u', 'd', 'w', 's', 'e', 'n']
ODD_SITE_LIST = ['d', 'u', 's', 'w', 'n', 'e']

__all__ = ['HubbardRegister']
class SiteRegister():
    """
    Class to handle the site names for the Hubbard defermionazed model.


    Parameters
    ----------
    xpos : int
        Position on the lattice along x
    ypos : int
        position on the lattice along y
    shape : tuple
        Shape of the lattice. For a 2x2 lattice it is (2, 2)
    """

    def __init__(self, xpos, ypos, shape, link_regs):

        name, str_list = self._init_site_register(xpos, ypos, shape)
        self.name = name
        self.is_even = (xpos+ypos)%2 == 0
        self._keys = str_list
        self.map = dict(zip(self._keys, range(len(str_list)) ))
        self.pos = (xpos, ypos)

        # It only has the matter inside
        self._qregister = QuantumRegister(2, f'q({xpos}, {ypos})')
        self._link_regs = link_regs

    def __repr__(self):
        """ Default class representation """
        return self.name

    def __len__(self):
        """ Number of qubits in the register """
        return len(self._keys)

    def __getitem__(self, key):
        """ Get access to the correct qubit index """
        if key not in self._keys:
            raise KeyError(f'{key} not in the available keys. Available keys are '+
                str(self._keys))

        # Access the matter
        if key in ('u', 'd'):
            qubit = self._qregister[self.map[key]]
        # Access the shared rishons
        else:
            qubit = self._link_regs[key][0]

        return qubit

    @property
    def is_boundary(self):
        """ Check if the site is a boundary site """
        if len(self._keys) < 6:
            is_boundary = True
        else:
            is_boundary = False
        return is_boundary

    @property
    def qregister(self):
        """ Quantum register property """
        return self._qregister

    @staticmethod
    def _init_site_register(xpos, ypos, shape):
        """
        Initialize a quantum register defined by the site
        in the positions xpos, ypos of a lattice

        Parameters
        ----------
        xpos : int
            Position on the lattice along x
        ypos : int
            position on the lattice along y
        shape : tuple
            Shape of the lattice. For a 2x2 lattice it is (2, 2)

        Return
        ------
        str
            The name of the site register, the string representation
            of its position
        list
            The ordered identifiers of its qubits following the naming
        """

        is_even = (xpos+ypos)%2 == 0
        if is_even:
            str_list = deepcopy(EVEN_SITE_LIST)
        else:
            str_list = deepcopy(ODD_SITE_LIST)

        if xpos == 0:
            str_list.remove('w')
        if xpos == shape[0]-1:
            str_list.remove('e')
        if ypos == 0:
            str_list.remove('s')
        if ypos == shape[1]-1:
            str_list.remove('n')

        name = f'q({xpos}, {ypos})'

        return name, str_list
class HubbardRegister():
    """
    Class to handle all the SiteRegisters of a Hubbard circuit, thus named
    HubbardRegister. It works exactly like a dictionary.
    Thus, by calling class['q(x, y)']['s'] you get the relative qubit, where
    x,y are the x,y position of the site you are interested in and 's' the
    species you are looking for. 's' is in ['u', 'd', 'e', 'w', 's', 'n']

    Parameters
    ----------
    shape: tuple of ints
        Shape of the Hubbard lattice
    ordering : list of ints
        New order of the sites to minimize the entanglement.
        Default is None, with the following ordering:
        - All the vertical links, left to right, down to up
        - All the horizontal links, left to right, down to up
        - All the sites, left to right, down to up
    """

    def __init__(self, shape, ordering=None):

        vert_links = [ii for ii in range(shape[0]*(shape[1]-1))]
        horiz_links = [ii for ii in range(shape[1]*(shape[0]-1))]
        links_qr = {}
        for link_idx in vert_links:
            links_qr[f'lv{link_idx}'] = QuantumRegister(1, f'lv{link_idx}')
        for link_idx in horiz_links:
            links_qr[f'lh{link_idx}'] = QuantumRegister(1, f'lh{link_idx}')

        self.qregisters = list(links_qr.values())
        self.registers = {}
        self.shape = shape
        for ypos in range(shape[1]):
            for xpos in range(shape[0]):

                link_regs = {}

                if xpos != 0:
                    link_regs['w'] = links_qr[f'lh{xpos-1+ypos*(shape[0]-1)}']

                if xpos != shape[0]-1:
                    link_regs['e'] = links_qr[f'lh{xpos+ypos*(shape[0]-1)}']

                if ypos != 0:
                    link_regs['s'] = links_qr[f'lv{ypos-1+xpos*(shape[1]-1)}']

                if ypos != shape[1]-1:
                    link_regs['n'] = links_qr[f'lv{ypos+xpos*(shape[1]-1)}']

                site = SiteRegister(xpos, ypos, shape, link_regs)

                self.registers[f'q({xpos}, {ypos})'] = site
                self.qregisters += [site.qregister]

        self.qregisters = np.array(self.qregisters, dtype=object)
        if ordering is not None:
            self.qregisters = self.qregisters[ordering]


    def __getitem__(self, key):
        """ Get access to the correct SiteRegister """

        return self.registers[key]

    def __len__(self):
        """ Number of SiteRegisters, i.e. dressed sites"""
        return len(self.registers)

    def keys(self):
        """ Identifiers of the SiteRegisters """
        return self.registers.keys()

    def values(self):
        """ SiteRegisters """
        return self.registers.values()

    def items(self):
        """ Keys and values of SiteRegisters """
        return self.registers.items()

    def __iter__(self):
        """ Iterate over the registers """
        return iter(self.registers)

