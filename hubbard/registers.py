from qiskit import QuantumRegister
from copy import deepcopy

EVEN_SITE_LIST = ['u', 'd', 'w', 's', 'e', 'n']
ODD_SITE_LIST = ['d', 'u', 's', 'w', 'n', 'e']

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

    def __init__(self, xpos, ypos, shape):

        name, str_list = self._init_site_register(xpos, ypos, shape)
        self.name = name
        self.is_even = (xpos+ypos)%2 == 0
        self._keys = str_list
        self.map = dict(zip(self._keys, range(len(str_list)) ))

        self._qregister = QuantumRegister(len(str_list), name)

    def __len__(self):
        """ Number of qubits in the register """
        return len(self._keys)

    def __getitem__(self, key):
        """ Get access to the correct qubit index """
        if key not in self._keys:
            raise KeyError(f'{key} not in the available keys. Available keys are '+
                str(self._keys))

        return self._qregister[self.map[key]]

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
