"""Defines a base interface for all callbacks and implements simple callbacks."""

import abc
import typing

if typing.TYPE_CHECKING:
    from src.population import Population


class Callback(abc.ABC):
    """Abstract base class used to build new callbacks.

    This class defines the general structure of the callbacks used.
    It's not required for a subclass to implement all the methods of
    this class (you can implement only those that will be useful for your case).

    Attributes:
        population (BasePopulation): Reference to the instance of a subclass of
            :class:`.Population` being evolved by one of `NEvoPy's`
            neuroevolutionary algorithms.
    """

    def set_population(self, population: "Population") -> None: ...

    def on_generation_start(
        self, current_generation: int, max_generations: int, **kwargs: typing.Any
    ) -> None:
        """Called at the beginning of each new generation.

        Subclasses should override this method for any actions to run.

        Args:
            current_generation (int): Number of the current generation.
            max_generations (int): Maximum number of generations.
        """

    def on_fitness_calculated(
        self, best_fitness: float, avg_fitness: float, **kwargs: typing.Any
    ) -> None:
        """Called right after the fitness values of the population's genomes
        are calculated.

        Subclasses should override this method for any actions to run.

        Args:
            best_fitness (float): Fitness of the fittest genome in the
                population.
            avg_fitness (float): Average fitness of the population's genomes.
        """

    def on_mass_extinction_counter_updated(
        self, mass_extinction_counter: int, **kwargs: typing.Any
    ) -> None:
        """Called right after the mass extinction counter is updated.

        Subclasses should override this method for any actions to run.

        Args:
            mass_extinction_counter (int): Current value of the mass extinction
                counter.
        """

    def on_mass_extinction_start(self, **kwargs: typing.Any) -> None:
        """Called at the beginning of a mass extinction event.

        Subclasses should override this method for any actions to run.

        Note:
            When this is called, :meth:`on_reproduction_start()` is usually
            not called (depends on the algorithm).
        """

    def on_reproduction_start(self, **kwargs: typing.Any) -> None:
        """Called at the beginning of the reproductive process.

        Subclasses should override this method for any actions to run.

        Note:
            When this is called, :meth:`on_mass_extinction_start()` is usually
            not called (depends on the algorithm).
        """

    def on_speciation_start(self, **kwargs: typing.Any) -> None:
        """Called at the beginning of the speciation process.

        Called after the reproduction or mass extinction have occurred and
        immediately before the speciation process. If the neuroevolution
        algorithm doesn't implement speciation, this method won't be called.

        Subclasses should override this method for any actions to run.
        """

    def on_generation_end(
        self, current_generation: int, max_generations: int, **kwargs: typing.Any
    ) -> None:
        """Called at the end of each generation.

        Subclasses should override this method for any actions to run.

        Args:
            current_generation (int): Number of the current generation.
            max_generations (int): Maximum number of generations.
        """

    def on_evolution_end(self, total_generations: int, **kwargs: typing.Any) -> None:
        """Called when the evolutionary process ends.

        Args:
            total_generations (int): Total number of generations processed
                during the evolutionary process. Might not be the maximum number
                of generations specified by the user, if some sort of early
                stopping occurs.
        """
