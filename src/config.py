import typing
import numpy as np
import pydantic

#: Name of the attributes whose values change according to the mass
#:  extinction counter (type: Tuple[float, float]).
MASS_EXTINCTION_MODIFIERS = {
    "weight_mutation_chance",
    "new_node_mutation_chance",
    "new_connection_mutation_chance",
    "enable_connection_mutation_chance",
    "weight_perturbation_pc",
    "weight_reset_chance",
}


def linear_activation(x: float) -> float:
    return x


def sigmoid(x: float, clip_value: int = 64) -> float:
    """Numeric stable implementation of the sigmoid function.

    Estimated lower-bound precision with a clip value of 64: 10^(-28).
    """
    x = np.clip(x, -clip_value, clip_value)
    return 1 / (1 + np.exp(-x))


def bool_function(x: float) -> float:
    if x > 0.5:
        return 1
    else:
        return 0


DEFAULT_ACTIVATION_FUNC = linear_activation


class NeatConfig(pydantic.BaseModel):
    out_nodes_activation: typing.Callable[[float], float] = sigmoid
    hidden_nodes_activation: typing.Callable[[float], float] = sigmoid
    bias_value: int = 1
    with_bias: bool = True
    # reproduction
    weak_genomes_removal_pc: float = 0.75
    weight_mutation_chance: tuple[float, float] = (0.7, 0.9)
    new_node_mutation_chance: tuple[float, float] = (0.1, 0.3)
    new_connection_mutation_chance: tuple[float, float] = (0.03, 0.3)
    enable_connection_mutation_chance: tuple[float, float] = (0.03, 0.3)
    disable_inherited_connection_chance: float = 0.75
    mating_chance: float = 0.7
    interspecies_mating_chance: float = 0.05
    rank_prob_dist_coefficient: float = 1.75
    # weight mutation specifics
    weight_perturbation_pc: tuple[float, float] = (0.1, 0.4)
    weight_reset_chance: tuple[float, float] = (0.1, 0.3)
    new_weight_interval: tuple[float, float] = (-2, 2)
    # mass extinction
    mass_extinction_threshold: int = 10
    maex_improvement_threshold_pc: float = 0.03
    # infanticide
    infanticide_output_nodes: bool = True
    infanticide_input_nodes: bool = True
    # random genomes
    random_genome_bonus_nodes: int = -2
    random_genome_bonus_connections: int = -2
    # genome distance coefficients
    excess_genes_coefficient: float = 1
    disjoint_genes_coefficient: float = 1
    weight_difference_coefficient: float = 0.5
    # speciation
    species_distance_threshold: float = 1
    species_elitism_threshold: float = 5
    species_no_improvement_limit: float = 15
    # others
    reset_innovations_period: int = 5
    allow_self_connections: bool = True
    initial_node_activation: int = 0

    _maex_cache: dict[str, float] = pydantic.PrivateAttr(default_factory=dict)
    _maex_counter: int = pydantic.PrivateAttr(default=0)

    def update_coefficients_for_extinction_counter(self, maex_counter: int) -> None:
        """Updates the mutation chances based on the current value of the mass
        extinction counter (generations without improvement).

        Args:
            maex_counter (int): Current value of the mass extinction counter
                (generations without improvement).
        """
        self._maex_counter = maex_counter
        for k in MASS_EXTINCTION_MODIFIERS:
            base_value, max_value = getattr(self, k)
            unit = (max_value - base_value) / self.mass_extinction_threshold
            self._maex_cache[k] = base_value + unit * maex_counter
