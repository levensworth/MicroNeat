import typing
from src.misc.callbacks.callback import Callback
from src.population import Population


class FitnessEarlyStopping(Callback):
    """Stops the evolution if a given fitness value is achieved.

    This callback is used to halt the evolutionary process when a certain
    fitness value is achieved by the population's best genome for a given number
    of consecutive generations.

    Args:
        fitness_threshold (float): Fitness to be achieved for the evolution to
            stop.
        min_consecutive_generations (int): Number of consecutive generations
            with a fitness equal or higher than ``fitness_threshold`` for the
            early stopping to occur.

    Attributes:
        fitness_threshold (float): Fitness to be achieved for the evolution to
            stop.
        min_consecutive_generations (int): Number of consecutive generations
            with a fitness equal or higher than ``fitness_threshold`` for the
            early stopping to occur.
        stopped_generation (Optional[int]): Generation in which the early
            stopping occurred. `None` if the early stopping never occurred.
    """

    def __init__(
        self, fitness_threshold: float, min_consecutive_generations: int
    ) -> None:
        super().__init__()
        self.fitness_threshold = fitness_threshold
        self.min_consecutive_generations = min_consecutive_generations
        self.stopped_generation: int | None = None
        self._consec_gens = 0
        self.population: Population | None = None

    def set_population(self, population: Population) -> None:
        self.population = population

    def on_fitness_calculated(
        self, best_fitness: float, avg_fitness: float, **kwargs: typing.Any
    ) -> None:
        if best_fitness >= self.fitness_threshold:
            self._consec_gens += 1
        else:
            self._consec_gens = 0

    def on_generation_end(
        self, current_generation: int, max_generations: int, **kwargs: typing.Any
    ) -> None:
        if self._consec_gens >= self.min_consecutive_generations:
            assert self.population
            self.population.stop_evolving = True
            self.stopped_generation = current_generation
