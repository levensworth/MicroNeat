import numpy as np
from src.genome import Genome
from src.population import Population
from src.visualization import visualize_genome


# genome = Genome.from_inputs_and_outputs(n_inputs=3, n_outputs=10)
# visualize_genome(genome)
# new_genome = genome.copy_with_random_weights()
# visualize_genome(new_genome)

# # plot_dag(genome.get_nodes())

# id_handler = IDHandler(num_inputs=3, num_outputs=10, has_bias=False)
# visualize_genome(genome)
# genome.add_random_hidden_node(id_handler)
# visualize_genome(genome)

# new_genome = genome.copy_with_random_weights()


# new_genome.add_random_connection(id_handler)
# visualize_genome(new_genome)
# print('here')

# child = genome.mate(new_genome)
# visualize_genome(child)

N_INPUTS = 3

def make_xor_data(num_variables: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """ Builds data using the `XOR` logic function.

    The generated inputs are all the possible combinations of input values with
    the specified number of variables. Each variable is a bit (0 or 1). The
    generated outputs are the results (a single bit each) of the `XOR` function
    applied to all the inputs.

    Example:

        >>> xor_in, xor_out = make_xor_data(num_variables=2)
        >>> for x, y in zip(xor_in, xor_out):
        ...     print(f"{x} -> {y}")
        ...
        [0 0] -> 0
        [0 1] -> 1
        [1 0] -> 1
        [1 1] -> 0

    Args:
        num_variables (int): Number of input variables for the `XOR` logic
            function.

    Returns:
        A tuple with two numpy arrays. The first array contains the input
        values, and the second array contains the output of the `XOR` function.
    """
    assert num_variables >= 2, "The XOR function needs at least 2 variables!"

    xor_inputs, xor_outputs = [], []
    for num in range(2 ** num_variables):
        binary = bin(num)[2:].zfill(num_variables)
        xin = [int(binary[0])]
        xout = int(binary[0])

        for bit in binary[1:]:
            xin.append(int(bit))
            xout ^= int(bit)

        xor_inputs.append(np.array(xin))
        xor_outputs.append(np.array(xout))

    return np.array(xor_inputs), np.array(xor_outputs)


# building the dataset
xor_inputs, xor_outputs = make_xor_data(N_INPUTS)

def fitness_function(genome: Genome, log=False):
    """ Implementation of the fitness function we're going to use.

    It simply feeds the XOR inputs to the given genome and calculates how well
    it did (based on the squared error).
    """
    # Shuffling the input, in order to prevent our networks from memorizing the
    # sequence of the answers.
    idx = np.random.permutation(len(xor_inputs))
    
    error = 0
    for x, y in zip(xor_inputs[idx], xor_outputs[idx]):
        # # Resetting the cached activations of the genome (optional).
        # genome.reset()

        # Feeding the input to the genome. A numpy array with the value 
        # predicted by the neural network is returned.
        h = genome.apply(x)[0]

        # Calculating the squared error.
        error += (y - h) ** 2

        if log:
            print(f"IN: {x}  |  OUT: {h:.4f}  |  TARGET: {y}")

    if log:
        print(f"\nError: {error}")

    return (1 / error) if error > 0 else 0

population = Population(
    size=200,
    n_inputs=N_INPUTS,
    n_outputs=1,
    with_bias=True
)


population.evolve(100, fitness_function)

for inp in xor_inputs:
    print(f'input: {inp}')
    print(population.fittest().apply(inp))

visualize_genome(population.fittest())

print(population.fittest().get_nodes())

# genome = Genome.from_inputs_and_outputs(2, 1)
# for _ in range(10):
#     genome.add_random_hidden_node(id_handler)
#     genome.add_random_connection(id_handler)

# total_con = 0
# for node in genome.get_nodes():
#     total_con += len(node.get_connections_in())

# print(total_con)
# print(len(genome.connections))


# visualize_genome(genome)
# result = genome.apply([1,0])
# print(result)
