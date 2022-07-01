# Defermionised Hubbard digital simulation

Description of what it actually means

## Table of content

- [The model](#model)
- [Initial state](#initial_state)
- [Visualize the state](#visualize_state)
- [Run the program](#run)
- [Measurable observables](#observables)
- [Plotting the results](#plot)

<div id='model'/>

### The model

After the transformation described above we end up with a lattice that has 2 qubits on
each site and 2 qubits on each link. The symmetries that we need to protect, using the
stabilisers formalism, are:

- Parity of the matter, in the vertexes of the lattice. The qubits on the
  vertexes can be either `00` or `11`. We will call qubits on the vertexes
  **matter** qubits.
- Parity of the links. The qubits on the links can be either `00` or `11`.
  From now on, we will call qubits on the links **rishons** qubits.
- Anticommutations rules of fermions, imposed by a symmetry on the plaquettes.

<div id='initial_state'/>

### Initial state

The initial state we select is the one for U>>t, and is basically
constituted by a chessboard-like state of the matter qubits. Thus, even vertexes will be
in the state `11` and odd vertexes in `00`.

<div id='visualize_state'/>

### Visualize the state

The python script let the user have a nice representation of the state. We show below an
example of the generated output:

```bash
┌───┐     ┌───┐
│0,0├──1──┤1,1│
└─┬─┘     └─┬─┘
  │         │
  1         1
  │         │
┌─┴─┐     ┌─┴─┐
│1,1├──1──┤0,0│
└───┘     └───┘
```

<div id='run'/>

## Run the program

The main executable of the repository is the file `main.py`. It enables the simulation of the Hubbard model using quantum circuits, by varying the onsite term U. It is possible to pass different parameters through command line to customize the simulation:

- `--shape`, Lattice shape, given as tuple. Default to `(2, 2)`. We suggest to use shapes elongated on the $x$ direction due to the way we compute the entanglement.
- `--dt`, Duration of a single timestep. Default to `0.1`.
- `--num_trotter_steps`, Number of trotter steps for the simulation of a single timestep. Default to `100`.
- `--t`, Hopping constant in the Hamiltonian. Default to `1`.
- `--U`, Onsite constant constant in the Hamiltonian.
  - If a list is provided, the onsite constant is generated as `np.linspace(U[0], U[1], U[2])`, so as `[low, high, num_steps]`.
  - If a string is provided, it should be the `PATH` to the file containing the values. Default to `[-8, -1/8, 100]`.

All these informations can be checked by command line using `python3 main.py -h`.

The results of the simulation are automatically stored in a folder called `data/idx`, where `idx` is an integer number uniquely characterizing the simulation. Inside the directory `data/idx/` the program will generate, during the runtime, six different files. The first one is a json file called `params.json`, where the input parameters of the simulation are saved. The others files contains the observables, described in the next section.

<div id='observables'/>

### Measurable observables

We measure the observables at each timestep. So, in the files,
each row will be a different observation.

- $\langle n_{\uparrow,\downarrow}\rangle$, the expectation value of the up and down spins on the sites. They are usually combined to get the charge and spin densities $\rho_{c,s}=\langle n_{\uparrow}\rangle \pm \langle n_{\downarrow}\rangle$. The results are recorded in the file `u_and_d.txt`. The first four columns represent the up species, while the last four the down species. The order is the same you obtain by calling the `HubbardRegister.keys()` method.
- $\langle n_{\uparrow}n_{\downarrow}\rangle$, the correlation between up and down spins on the sites. The results are recorded in the file `ud.txt`.
- $\langle K\rangle$, the expectation value of the kinetic term, which is equivalent to the hopping part of the Hamiltonian. The results are recorded in the file `kinetic.txt`.
- Checks on the symmetry of the plaquettes. If a `1` is present, then the symmetry is not respected and the evolution is worthless. There is a column for each plaquette. The results are recorded in the file `symmetry_check.txt`.
- $S_V$, the Von Neumann entanglement entropy by cutting in half the system. The results are recorded in the file `entanglement.txt`. The cut is done horizontally. For the implementation procedure, the cut is exactly half of the system only if the system has shape $(x, 2)$. We represent below the example of a cut in a $(3,2)$ lattice.

```bash
  O─r─O─r─O
  |   |   |
  | ┌───┐ |
  r │ r │ r
 ───┘ | └───
  |   |   |
  O─r─O─r─O
```

<div id='plot'/>

### Plotting the results

We also provide a script to plot (and optionally save) the results of a simulation in a standard way.
The script is `plot.py` and takes the following parameters:

- `plot_index`, a non-optional parameter that identifies the simulation you want to inspect.
- `--save`, if provided save the results in the `.pdf` format.
- `--path`, If provided, save in this PATH. Otherwise, save in `data/idx/`.

The plotting procedures are available for the kinetic, charge/spin densities and the entanglement.
Please feel free to implement plotting of the other quantities in `hubbard/plotter.py`, following
the examples already written, and then add the new lines in `plot.py`.

> :warning: If the plaquette symmetry is not satisfied at *any* time-step the plotter script will raise a `RuntimeError`.
