import numpy as np
from tqdm import tqdm
import typing
from src import utils
from src.config import NeatConfig
from src.genes import NodeGene
from src.genome import Genome
from src.id_handler import IDHandler
from src.schedulers.pool_scheduler import PoolProcessingScheduler
from src.species import NeatSpecies


class Population:
    _DEFAULT_CONFIG = NeatConfig

    def __init__(
        self,
        size: int,
        n_inputs: int,
        n_outputs: int,
        with_bias: bool = False,
        config: NeatConfig | None = None,
    ) -> None:
        # Set the size of the population
        self._size = size
        # set the configuration
        if config:
            self._config = config
        else:
            self._config = self._DEFAULT_CONFIG()

        self._base_genome = Genome.from_inputs_and_outputs(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            with_bias=with_bias,
            config=self._config,
        )

        self._id_handler = IDHandler(
            num_inputs=self._base_genome.get_input_shape(),
            num_outputs=self._base_genome.get_output_shape(),
            has_bias=with_bias,
        )

        self._cached_rank_prob_dist = utils.rank_prob_dist(
            size=size, coefficient=self._config.rank_prob_dist_coefficient
        )

        # creating initial genomes
        self.genomes = [
            self._base_genome.copy_with_random_weights() for _ in range(size)
        ]

        new_sp = NeatSpecies(
            species_id=self._id_handler.next_species_id(), generation=0
        )
        new_sp.add_members(self.genomes)
        new_sp.set_random_representative()
        self.species = {new_sp.id: new_sp}

        self._scheduler = PoolProcessingScheduler(
            num_processes=5,
        )

        self.record_holder = self._base_genome

    def evolve(
        self,
        generations: int,
        fitness_function: typing.Callable[[Genome], float],
    ) -> None:
        self._last_improvement = 0
        self._past_best_fitness = float("-inf")

        # evolving
        self.stop_evolving = False
        generation_num = 0
        for generation_num in range(generations):
            # calculating fitness
            fitness_results = []
            for gen in tqdm(self.genomes):
                fitness_results.append(fitness_function(gen))

            # fitness_results = self._scheduler.run(self.genomes, func=fitness_function)

            # assigning fitness and adjusted fitness
            for genome, fitness in zip(self.genomes, fitness_results):
                genome.fitness = fitness
                assert genome.species_id is not None
                sp = self.species[genome.species_id]
                genome.adj_fitness = genome.fitness / len(sp.members)
            best = self.fittest()

            self.record_holder = (
                best
                if best.fitness > self.record_holder.fitness
                else self.record_holder
            )

            # counting max number of hidden nodes in one genome
            self.__max_hidden_nodes = np.max(
                [len(g.hidden_nodes) for g in self.genomes]
            )

            # counting max number of hidden connections in one genome
            self.__max_hidden_connections = np.max(
                [
                    len(
                        [
                            c
                            for c in g.connections
                            if (
                                c.is_enabled
                                and (
                                    c.get_source_node().get_type()
                                    == NodeGene.NodeTypeEnum.HIDDEN
                                    or c.get_destination_node().get_type()
                                    == NodeGene.NodeTypeEnum.HIDDEN
                                )
                            )
                        ]
                    )
                    for g in self.genomes
                ]
            )

            # callback: on_fitness_calculated
            avg_fitness = self.average_fitness()

            # checking improvements
            improv_diff = best.fitness - self._past_best_fitness
            improv_min_pc = self._config.maex_improvement_threshold_pc
            if improv_diff >= abs(self._past_best_fitness * improv_min_pc):
                self._mass_extinction_counter = 0
                self._past_best_fitness = best.fitness
            else:
                self._mass_extinction_counter += 1

            self._config.update_coefficients_for_extinction_counter(
                self._mass_extinction_counter
            )

            # checking mass extinction
            if self._mass_extinction_counter >= self._config.mass_extinction_threshold:
                # mass extinction
                self._mass_extinction_counter = 0
                self.genomes = [best] + [
                    self._random_genome_with_extras() for _ in range(self._size - 1)
                ]
                print("mass extinction")
                assert len(self.genomes) == self._size
            else:
                # reproduction
                self.reproduction()

            # speciation
            self.speciation(generation=generation_num)

            # early stopping
            if self.stop_evolving:
                break

    def reproduction(self) -> None:
        new_pop = []

        # elitism
        for sp in self.species.values():
            sp.members.sort(key=lambda genome: genome.fitness, reverse=True)

            # preserving the most fit individual
            if len(sp.members) >= self._config.species_elitism_threshold:
                new_pop.append(sp.members[0])

            # removing the least fit individuals
            r = int(len(sp.members) * self._config.weak_genomes_removal_pc)
            if 0 < r < len(sp.members):
                r = len(sp.members) - r
                for g in sp.members[r:]:
                    self.genomes.remove(g)
                sp.members = sp.members[:r]

        # calculating the number of children for each species
        offspring_count = self.offspring_proportion(
            num_offspring=self._size - len(new_pop)
        )

        # creating new genomes
        self._invalid_genomes_replaced = 0
        for sp in self.species.values():
            # reproduction probabilities (rank-based selection)
            prob = self._cached_rank_prob_dist[: len(sp.members)]
            prob_sum = np.sum(prob)

            if abs(prob_sum - 1) > 1e-8:
                # normalizing distribution
                prob = prob / prob_sum

            # generating offspring
            babies = [
                self.generate_offspring(species=sp, rank_prob_dist=prob)  # type: ignore
                for _ in range(offspring_count.get(sp.id, 0))
            ]
            new_pop += babies

        assert len(new_pop) == self._size
        self.genomes = new_pop

        # # checking if the innovation ids should be reset
        # reset_period = self._config.reset_innovations_period
        # reset_counter = self._id_handler.reset_counter
        # if reset_period is not None and reset_counter > reset_period:
        #     self._id_handler.reset()
        # self._id_handler.reset_counter += 1

    def offspring_proportion(self, num_offspring: int) -> dict[int, int]:
        """Calculates the number of descendants each species will leave for the
        next generation.

        Every species is assigned a potentially different number of offspring in
        proportion to the sum of adjusted fitnesses of its member organisms
        :cite:`stanley:ec02`. This selection method is called `roulette wheel
        selection`.

        Args:
            num_offspring (int): Number of genomes to be generated by all the
                species combined.

        Returns
            A dictionary mapping the ID of each of the population's species to
            the number of descendants it will leave for the next generation.
        """
        adj_fitness = {sp.id: sp.avg_fitness() for sp in self.species.values()}
        total_adj_fitness = np.sum(list(adj_fitness.values()))

        offspring_count = {}
        count = num_offspring

        if total_adj_fitness > 0:
            for sid in self.species:
                v = int(num_offspring * adj_fitness[sid] / total_adj_fitness)
                offspring_count[sid] = v
                count -= offspring_count[sid]

        for _ in range(count):
            sid = utils.select_choice(list(self.species.keys()))
            if sid not in offspring_count:
                offspring_count[sid] = 0
            offspring_count[sid] += 1

        assert np.sum(list(offspring_count.values())) == num_offspring
        return offspring_count

    def generate_offspring(
        self, species: NeatSpecies, rank_prob_dist: typing.Sequence
    ) -> Genome:
        """Generates a new genome from one or more genomes of the species.

        The offspring can be generated either by mating two randomly chosen
        genomes (sexual reproduction) or by cloning a single genome (asexual
        reproduction / binary fission). After the newly born genome is created,
        it has a chance of mutating. The possible mutations are:

            | . Enabling a disabled connection;
            | . Changing the weights of one or more connections;
            | . Creating a new connection between two random nodes;
            | . Creating a new random hidden node.

        Args:
            species (NeatSpecies): Species from which the offspring will be
                generated.
            rank_prob_dist (Sequence): Sequence (usually a numpy array)
                containing the chances of each of the species genomes being the
                first parent of the newborn genome.

        Returns:
            A newly generated genome.
        """

        g1: Genome = utils.select_choice(species.members, p=rank_prob_dist)

        # mating / cross-over
        if utils.chance(self._config.mating_chance):
            # interspecific
            if len(self.species) > 1 and utils.chance(
                self._config.interspecies_mating_chance
            ):
                g2 = utils.select_choice(
                    [g for g in self.genomes if g.species_id != species.id]
                )
            # intraspecific
            else:
                g2: Genome = utils.select_choice(species.members)  # type: ignore
            baby = g1.mate(g2)
        # binary_fission
        else:
            baby = g1.clone()

        # enable connection mutation
        if utils.chance(self._config._maex_cache["enable_connection_mutation_chance"]):
            baby.enable_random_connection()

        # weight mutation
        if utils.chance(self._config._maex_cache["weight_mutation_chance"]):
            baby.mutate_random_weight()

        # new connection mutation
        if utils.chance(self._config._maex_cache["new_connection_mutation_chance"]):
            baby.add_random_connection(self._id_handler)

        # new node mutation
        if utils.chance(self._config._maex_cache["new_node_mutation_chance"]):
            baby.add_random_hidden_node(self._id_handler)

        # checking genome validity
        valid_out = not self._config.infanticide_output_nodes or baby.valid_out_nodes()
        valid_in = not self._config.infanticide_input_nodes or baby.valid_in_nodes()

        # genome is valid
        if valid_out and valid_in:
            return baby

        # invalid genome: replacing with a new random genome
        self._invalid_genomes_replaced += 1
        return self._random_genome_with_extras()

    def speciation(self, generation: int) -> None:
        """Divides the population's genomes into species.

        The importance of speciation for NEAT:

        "Speciating the population allows organisms to compete primarily within
        their own niches instead of with the population at large. This way,
        topological innovations are protected in a new niche where they have
        time to optimize their structure through competition within the niche.
        The idea is to divide the population into species such that similar
        topologies are in the same species." - :cite:`stanley:ec02`

        The distance (compatibility) between a pair of genomes is calculated
        based on to the number of excess and disjoint genes between them. See
        :meth:`.NeatGenome.distance()` for more information.

        About the speciation process:

        "Each existing species is represented by a random genome inside the
        species from the previous generation. A given genome g in the current
        generation is placed in the first species in which g is compatible with
        the representative genome of that species. This way, species do not
        overlap. If g is not compatible with any existing species, a new species
        is created with g as its representative." - :cite:`stanley:ec02`

        Species that haven't improved their fitness for a pre-defined number of
        generations are extinct, i.e., they are removed from the population
        and aren't considered for the speciation process. This number is
        configurable.

        Args:
            generation (int): Current generation number.
        """
        extinction_threshold = self._config.species_no_improvement_limit

        # checking improvements and resetting members
        removed_sids = []
        for sp in self.species.values():
            past_best_fitness = sp.best_fitness
            sp.best_fitness = sp.fittest().fitness

            if past_best_fitness is not None:
                if sp.best_fitness > past_best_fitness:
                    # updating improvement record
                    sp.last_improvement = generation
                elif (generation - sp.last_improvement) > extinction_threshold:
                    # marking species for extinction (it hasn't shown
                    # improvements in the past few generations)
                    removed_sids.append(sp.id)

            # resetting members
            sp.members = []

        # extinction of unfit species
        for sid in removed_sids:
            self.species.pop(sid)

        # assigning genomes to species
        dist_threshold = self._config.species_distance_threshold
        for genome in self.genomes:
            chosen_species = None

            # checking compatibility with existing species
            for sp in self.species.values():
                assert sp.representative
                if genome.distance(sp.representative) <= dist_threshold:
                    chosen_species = sp
                    break

            # creating a new species, if needed
            if chosen_species is None:
                sid = self._id_handler.next_species_id()
                chosen_species = NeatSpecies(species_id=sid, generation=generation)
                chosen_species.representative = genome
                self.species[chosen_species.id] = chosen_species

            # adding genome to species
            chosen_species.members.append(genome)
            genome.species_id = chosen_species.id

        # deleting empty species and updating representatives
        for sp in list(self.species.values()):
            if len(sp.members) == 0:
                self.species.pop(sp.id)
            else:
                sp.set_random_representative()

    def fittest(self) -> Genome:
        """Returns the most fit genome in the population."""
        return self.genomes[int(np.argmax([g.fitness for g in self.genomes]))]

    def average_fitness(self) -> float:
        """Returns the average fitness of the population's genomes."""
        return np.mean([g.fitness for g in self.genomes])  # type: ignore

    def _random_genome_with_extras(self) -> Genome:
        """Creates a new random genome with extra hidden nodes and connections.

        The number of hidden nodes in the new genome will be randomly picked
        from the interval `[0, max_hn + bonus_hn]`, where `max_hn` is the
        number of hidden nodes in the genome (of the population) with the
        greatest number of hidden nodes and `bonus_hn` is a bonus value
        specified in the settings. The number of hidden connections in the new
        genome is chosen in a similar way.

        Returns:
            A new random genome with extra hidden nodes and connections.
        """
        new_genome = self._base_genome.copy_with_random_weights()

        # adding hidden nodes
        max_hnodes = self.__max_hidden_nodes + self._config.random_genome_bonus_nodes
        if max_hnodes > 0:
            for _ in range(np.random.randint(low=0, high=(max_hnodes + 1))):
                new_genome.add_random_hidden_node(self._id_handler)

        # adding random connections
        max_hcons = (
            self.__max_hidden_connections + self._config.random_genome_bonus_connections
        )
        if max_hcons > 0:
            for _ in range(np.random.randint(low=0, high=(max_hcons + 1))):
                new_genome.add_random_connection(self._id_handler)

        return new_genome
