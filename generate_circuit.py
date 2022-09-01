from hubbard import generate_evolution_circuit

if __name__ == '__main__':
    shapes = [(2, 2), (2, 3), (4, 3), (4, 4)]

    for shape in shapes:
        time_step = 0.01
        num_trotter_steps = 1
        hopping_constant = 1
        filename = None

        generate_evolution_circuit(shape,
                                    time_step,
                                    num_trotter_steps,
                                    hopping_constant,
                                    filename)