from qiskit import QuantumRegister
from copy import deepcopy
from collections import namedtuple
import numpy as np
from ..qiskit.registers import SiteRegister

EVEN_SITE_LIST = ['u', 'd', 'w', 's', 'e', 'n']
ODD_SITE_LIST = ['d', 'u', 's', 'w', 'n', 'e']

class NewSiteRegister(SiteRegister):

    def __init__(self, xpos, ypos, shape, link_regs):
        super().__init__(xpos, ypos, shape)

        # It only has the matter inside
        self._qregister = QuantumRegister(2, f'q({xpos}, {ypos})')
        self._link_regs = link_regs

        #self.qregisters = list(link_regs.values() ) + [self._qregister]

    def __getitem__(self, key):
        """ Get access to the correct qubit index """
        if key not in self._keys:
            raise KeyError(f'{key} not in the available keys. Available keys are '+
                str(self._keys))

        if key in ('u', 'd'):
            qubit = self._qregister[self.map[key]]
        else:
            qubit = self._link_regs[key][0]

        return qubit


class HubbardRegister():

    def __init__(self, shape):

        avail_links = [(ii, jj) for ii in range(shape[0]-1) for jj in range(shape[1]+1)]
        avail_links += [(shape[0]-1, jj) for jj in range(shape[1]) if jj%2==1]
        links_qr = {}
        for link_idx in avail_links:
            links_qr[f'l{link_idx}'] = QuantumRegister(1, f'l{link_idx}')

        self.qregisters = list(links_qr.values())
        self.registers = {}
        for xpos in range(shape[0]):
            for ypos in range(shape[1]):

                link_regs = {}

                if xpos != 0:
                    link_regs['w'] = links_qr[f'l({xpos-1}, {ypos*2})']

                if xpos != shape[0]-1:
                    link_regs['e'] = links_qr[f'l({xpos}, {ypos*2})']

                if ypos != 0:
                    link_regs['s'] = links_qr[f'l({xpos}, {ypos*2-1})']

                if ypos != shape[1]-1:
                    link_regs['n'] = links_qr[f'l({xpos}, {ypos*2+1})']

                site = NewSiteRegister(xpos, ypos, shape, link_regs)

                self.registers[f'q({xpos}, {ypos})'] = site
                self.qregisters += [site.qregister]

    def __getitem__(self, key):
        """ Get access to the correct qubit index """

        return self.registers[key]

