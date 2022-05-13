# Defermionised Hubbard digital simulation

Description of what it actually means

## The model

After the transformation described above we end up with a lattice that has 2 qubits on
each site and 2 qubits on each link. The symmetries that we need to protect, using the
stabilisers formalism, are:

- Parity of the matter, in the vertexes of the lattice. The qubits on the
  vertexes can be either `00` or `11`. We will call qubits on the vertexes
  **matter** qubits.
- Parity of the links. The qubits on the links can be either `00` or `11`.
  From now on, we will call qubits on the links **rishons** qubits.
- Anticommutations rules of fermions, imposed by a symmetry on the plaquettes.

## Initial state

The initial state we select is the one for *insert parametric regime*, and is basically
constituted by a chessboard-like state of the matter qubits. Thus, even vertexes will be
in the state `11` and odd vertexes in `00`.

## Visualize the state

The python script let the user have a nice representation of the state. We show below an
example of the generated output:

┌───┐         ┌───┐
│0,0├───1─1───┤1,1│
└─┬─┘         └─┬─┘
  │             │
  0             1
  │             │
  0             1
  │             │
┌─┴─┐         ┌─┴─┐
│1,1├───1─1───┤0,0│
└───┘         └───┘
