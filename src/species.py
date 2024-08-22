import numpy as np

from src.genome import Genome


class NeatSpecies:
    """ Represents a species within NEAT's evolutionary environment.

    Args:
        species_id (int): Unique identifier of the species.
        generation (int): Current generation. The generation in which the
            species is born.

    Attributes:
        representative (Optional[NeatGenome]): Genome used to represent the
            species.
        members (List[NeatGenome]): List with the genomes that belong to the
            species.
        last_improvement (int): Generation in which the species last showed
            improvement of its fitness. The species fitness in a given
            generation is equal to the fitness of the species most fit genome on
            that generation.
        best_fitness (Optional[float]): The last calculated fitness of the
            species most fit genome.
    """

    def __init__(self, species_id: int, generation: int) -> None:
        self._id = species_id
        self.representative: Genome | None = None 
        self.members: list[Genome] = []

        self._creation_gen = generation
        self.last_improvement = generation
        self.best_fitness: float | None = None 

    @property
    def id(self) -> int:
        """ Unique identifier of the species. """
        return self._id

    def set_random_representative(self) -> None:
        """ Randomly chooses a new representative for the species. """
        self.representative = np.random.choice(self.members) # type: ignore

    def avg_fitness(self) -> float:
        """ Returns the average fitness of the species genomes. """
        return float(np.mean([g.fitness for g in self.members]))

    def fittest(self) -> Genome:
        """ Returns the fittest member of the species. """
        return self.members[int(np.argmax([g.fitness for g in self.members]))]
    

    def add_members(self, genomes: list[Genome]) -> None:
        self.members = genomes[:]
        for idx, gen in enumerate(genomes):
            gen.species_id = self.id
            self.members[idx].species_id = self.id

        
