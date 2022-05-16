import numpy as np

from ..qiskit.registers import SiteRegister
from .map_1d import quasi_1d_chain


class QCSiteRegister(SiteRegister):
    """
    Version of the SiteRegister for the MPS
    simulation. Get the register name by simply
    using the class, and the relative index using
    the square brackets

    Available 1d mappings:
        - quasi_1d_chain

    Parameters
    ----------
    xpos : int
        Position on the lattice along x
    ypos : int
        position on the lattice along y
    shape : tuple
        Shape of the lattice. For a 2x2 lattice it is (2, 2)
    mapping_1d : str
        Way of mapping the 2d hubbard system to a 1-dimensional
        system
    """

    mapping_1d = {
        'quasi_1d_chain' : quasi_1d_chain
    }

    def __init__(self, xpos, ypos, shape, mapping_1d):

        super().__init__(xpos, ypos, shape)

        map_to_1d = self.mapping_1d[mapping_1d](shape)
        self.map_to_1d = np.array([ name+species for name, species in map_to_1d] )

    def __getitem__(self, key):
        """ Get access to the correct qubit index """
        if key not in self._keys:
            raise KeyError(f'{key} not in the available keys. Available keys are '+
                str(self._keys))

        idx = self.name + key
        value = np.nonzero(self.map_to_1d == idx)[0][0]

        return value
