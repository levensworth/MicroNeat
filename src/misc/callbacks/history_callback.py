
import typing
from src.misc.callbacks.callback import Callback
from timeit import default_timer

from rich.progress import Progress
if typing.TYPE_CHECKING:
    from src.population import Population 

class History(Callback):
    """ Callback that records events during the evolutionary process.

    Besides the regular attributes in the methods signature, the caller can also
    pass other attributes through "kwargs". All the attributes passed to the
    methods will have they value stored in the :attr:`.history` dictionary.

    Attributes:
        history (Dict[str, List[Any]]): Dictionary that maps an attribute's name
            to a list with the attribute's values along the evolutionary
            process.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: dict[str, list[float | int | str]] = {
            "best_fitness": [],
            "avg_fitness": [],
            "mass_extinction_counter": [],
            "weight_mutation_chance": [],
            "weight_perturbation_pc": [],
            "mass_extinction_events": [],
            "processing_time": [],
        }

        self._current_generation = 0
        self._current_generation_processing = Progress(transient=True)
        self._task_id = self._current_generation_processing.add_task('fitness_eval')
        self._timer = 0.0

    def __getattr__(self, key: str) -> list[float | int | str]:
        return self.history[key]

    def _update_history(self, **kwargs: float | int | str) -> None:
        for k, v in kwargs.items():
            if k not in self.history:
                self.history[k] = []
            self.history[k].append(v)

    def set_population(self, population: 'Population') -> None:
        self.population = population

    def on_generation_start(self,
                            current_generation: int,
                            max_generations: int,
                            **kwargs: typing.Any) -> None:
        self._timer = default_timer()
        self._update_history(**kwargs)
        self._current_generation = current_generation

    def on_fitness_calculated(self,
                              best_fitness: float,
                              avg_fitness: float,
                              **kwargs: typing.Any) -> None:
        self._update_history(best_fitness=best_fitness,
                             avg_fitness=avg_fitness,
                             **kwargs)

    def on_mass_extinction_counter_updated(self,
                                           mass_extinction_counter: int,
                                           **kwargs: typing.Any) -> None:
        weight_mutation_chance = self.population._config._maex_cache['weight_mutation_chance']
        weight_perturbation_pc = self.population._config._maex_cache['weight_perturbation_pc']

        self._update_history(
            mass_extinction_counter=mass_extinction_counter,
            weight_mutation_chance=weight_mutation_chance,
            weight_perturbation_pc=weight_perturbation_pc,
            **kwargs,
        )

    def on_mass_extinction_start(self, **kwargs: typing.Any) -> None:
        self._update_history(mass_extinction_events=self._current_generation,
                             **kwargs)

    def on_reproduction_start(self, **kwargs: typing.Any) -> None:
        self._update_history(**kwargs)

    def on_speciation_start(self, **kwargs: typing.Any) -> None:
        self._update_history(**kwargs)

    def on_generation_end(self,
                          current_generation: int,
                          max_generations: int,
                          **kwargs: typing.Any) -> None:
        self._update_history(processing_time=default_timer() - self._timer,
                             **kwargs)

    def on_evolution_end(self, total_generations: int, **kwargs: typing.Any) -> None:
        self._update_history(**kwargs)
