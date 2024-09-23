from copy import copy
import io
import pathlib
import pickle
import typing
import numpy as np
from src import utils
from src.config import DEFAULT_ACTIVATION_FUNC, NeatConfig
from src.genes import ConnectionGene, NodeGene, align_connections
from src.id_handler import IDHandler
from src.misc.serialization import pickle_load, pickle_save
from src.nn import NeuralNetwork


class Genome:
    def __init__(
        self,
        input_nodes: list[NodeGene],
        output_nodes: list[NodeGene],
        hidden_nodes: list[NodeGene],
        connections: list[ConnectionGene],
        config: NeatConfig,
        bias_node: NodeGene | None = None,
    ) -> None:
        self.species_id: int | None = None

        self.fitness = 0.0
        self.adj_fitness = 0.0

        self.input_nodes: list[NodeGene] = input_nodes
        self.hidden_nodes: list[NodeGene] = hidden_nodes
        self.output_nodes: list[NodeGene] = output_nodes
        self.bias_node = bias_node
        self.connections: list[ConnectionGene] = connections

        self._config = config

    def __repr__(self) -> str:
        return f"Genome(n_inputs = {len(self.input_nodes)} | n_hidden = {len(self.hidden_nodes)} | n_output = {len(self.output_nodes)} | n_cons = {len(self.connections)})"

    @classmethod
    def _create_input_nodes(cls, n_inputs: int, config: NeatConfig) -> list[NodeGene]:
        input_nodes: list[NodeGene] = []
        node_counter = 0
        for _ in range(n_inputs):
            input_nodes.append(
                NodeGene(
                    node_id=node_counter,
                    node_type=NodeGene.NodeTypeEnum.INPUT,
                    activation_func=DEFAULT_ACTIVATION_FUNC,
                )
            )
            node_counter += 1

        return input_nodes

    @classmethod
    def _create_output_nodes(
        cls,
        n_outputs: int,
        input_nodes: list[NodeGene],
        bias_node: NodeGene | None,
        config: NeatConfig,
    ) -> list[NodeGene]:
        node_counter = len(input_nodes)
        node_counter += 1 if bias_node is not None else 0
        output_nodes: list[NodeGene] = []

        for _ in range(n_outputs):
            output_node = NodeGene(
                node_id=node_counter,
                node_type=NodeGene.NodeTypeEnum.OUTPUT,
                activation_func=config.out_nodes_activation,
            )
            output_nodes.append(output_node)
            node_counter += 1
        return output_nodes

    @classmethod
    def _create_connections(
        cls,
        input_nodes: list[NodeGene],
        output_nodes: list[NodeGene],
        config: NeatConfig,
    ) -> list[ConnectionGene]:
        connection_counter = 0
        connections: list[ConnectionGene] = []
        # add all connections
        for output_node in output_nodes:
            for src_node in input_nodes:
                conn = cls._create_connection_with_random_weights(
                    connection_id=connection_counter,
                    src_node=src_node,
                    dst_node=output_node,
                    config=config,
                )
                connections.append(conn)
                connection_counter += 1
        return connections

    @classmethod
    def from_inputs_and_outputs(
        cls, n_inputs: int, n_outputs: int, config: NeatConfig, with_bias: bool = False
    ) -> "Genome":
        input_nodes: list[NodeGene] = cls._create_input_nodes(n_inputs, config=config)
        if with_bias:
            bias = NodeGene(
                node_id=len(input_nodes),
                node_type=NodeGene.NodeTypeEnum.BIAS,
                activation_func=DEFAULT_ACTIVATION_FUNC,
            )
        else:
            bias = None

        hidden_nodes: list[NodeGene] = []
        output_nodes: list[NodeGene] = cls._create_output_nodes(
            n_outputs, input_nodes=input_nodes, bias_node=bias, config=config
        )

        connection_sources = input_nodes + ([bias] if bias is not None else [])
        connections: list[ConnectionGene] = cls._create_connections(
            connection_sources, output_nodes, config=config
        )

        return Genome(
            input_nodes,
            output_nodes,
            hidden_nodes,
            connections,
            config=config,
            bias_node=bias,
        )

    @classmethod
    def from_connections(
        cls, connections: list[ConnectionGene], config: NeatConfig
    ) -> "Genome":
        input_nodes_dict: dict[int, NodeGene] = {}
        hidden_nodes_dict: dict[int, NodeGene] = {}
        output_nodes_dict: dict[int, NodeGene] = {}
        bias_node = None

        node_lookup_table: dict[int, NodeGene] = {}
        # extract all nodes
        for con in connections:
            for node in [con.get_source_node(), con.get_destination_node()]:
                new_node = node.copy()
                node_lookup_table[new_node.get_id()] = new_node
                match node.get_type():
                    case NodeGene.NodeTypeEnum.INPUT:
                        input_nodes_dict[node.get_id()] = new_node
                    case NodeGene.NodeTypeEnum.OUTPUT:
                        output_nodes_dict[node.get_id()] = new_node
                    case NodeGene.NodeTypeEnum.HIDDEN:
                        hidden_nodes_dict[node.get_id()] = new_node
                    case NodeGene.NodeTypeEnum.BIAS:
                        bias_node = new_node
                    case _:
                        raise ValueError(f"type = {node.get_type()} is not supported")

        new_connections = []
        # re populate connections
        for con in connections:
            try:
                new_connections.append(
                    cls._create_connection(
                        connection_id=con.get_id(),
                        src_node=node_lookup_table[con.get_source_node().get_id()],
                        dst_node=node_lookup_table[con.get_destination_node().get_id()],
                        weight=con.weight,
                        is_enabled=con.is_enabled,
                    )
                )
            except ValueError:
                # this is due to the same connection
                # existing in different genes
                pass

        return Genome(
            input_nodes=list(input_nodes_dict.values()),
            output_nodes=list(output_nodes_dict.values()),
            hidden_nodes=list(hidden_nodes_dict.values()),
            connections=new_connections,
            config=config,
            bias_node=bias_node,
        )

    def apply(self, inputs: list[float]) -> list[float]:
        nn = NeuralNetwork(self)
        return nn.process(inputs)

    def get_input_shape(self) -> int:
        return len(self.input_nodes)

    def get_output_shape(self) -> int:
        return len(self.output_nodes)

    def get_nodes(self) -> list[NodeGene]:
        result = []
        result += self.input_nodes
        result += self.hidden_nodes
        result += self.output_nodes
        result += [self.bias_node] if self.bias_node else []
        return result

    def mate(self, other: "Genome") -> "Genome":
        ordered_genes = align_connections(self.connections, other.connections)

        chosen_connections: list[ConnectionGene] = []

        for c1, c2 in zip(*ordered_genes):
            if c1 is None and self.adj_fitness > other.adj_fitness:
                # case 1: the gene is missing on self and self is dominant
                # (higher fitness); action: ignore the gene
                continue
            if c2 is None and self.adj_fitness < other.adj_fitness:
                # case 2: the gene is missing on other and other is dominant
                # (higher fitness); action: ignore the gene
                continue

            # case 3: the gene is missing either on self or on other and their
            # fitness are equal; action: random choice

            # case 4: the gene is present both on self and on other; action:
            # random choice

            c: ConnectionGene | None = np.random.choice((c1, c2))  # type: ignore
            if c is not None:
                # if the gene is disabled in either parent, it has a chance to
                # also be disabled in the new genome
                enabled = True
                if (c1 is not None and not c1.is_enabled) or (
                    c2 is not None and not c2.is_enabled
                ):
                    enabled = not utils.chance(
                        self._config.disable_inherited_connection_chance
                    )

                copied_connection = copy(
                    c
                )  # this is a shallow copy because I just don't want to overwrite the enablement
                copied_connection.is_enabled = enabled
                chosen_connections.append(copied_connection)

        return self.from_connections(chosen_connections, config=self._config)

    def mutate_random_weight(self) -> None:
        for connection in self.connections:
            # TODO: check if maybe introduce more change
            if utils.chance(self._config._maex_cache["weight_reset_chance"]):
                # perturbing the connection
                connection.weight = np.random.uniform(*self._config.new_weight_interval)
            else:
                # resetting the connection
                p = np.random.uniform(
                    low=-self._config.weight_perturbation_pc[0],
                    high=self._config.weight_perturbation_pc[1],
                )
                d = connection.weight * p
                connection.weight += d

    def enable_random_connection(self) -> None:
        """Randomly activates a disabled connection gene."""
        disabled = [c for c in self.connections if not c.is_enabled]
        if len(disabled) > 0:
            connection: ConnectionGene = np.random.choice(disabled)  # type: ignore
            connection.is_enabled = True

    def add_random_connection(
        self, id_handler: IDHandler
    ) -> tuple[NodeGene, NodeGene] | None:
        all_src_nodes = self.get_nodes()
        np.random.shuffle(all_src_nodes)  # type: ignore

        all_dest_nodes = [
            n
            for n in all_src_nodes
            if (
                n.get_type() != NodeGene.NodeTypeEnum.BIAS
                and n.get_type() != NodeGene.NodeTypeEnum.INPUT
            )
        ]
        np.random.shuffle(all_dest_nodes)  # type: ignore

        for src_node in all_src_nodes:
            for dest_node in all_dest_nodes:
                if src_node.get_id() != dest_node.get_id():
                    if not self.does_connection_exists(src_node, dest_node):
                        cid = id_handler.get_next_connection_id(
                            src_node.get_id(), dest_node.get_id()
                        )
                        self.connections.append(
                            self._create_connection_with_random_weights(
                                cid, src_node, dest_node, config=self._config
                            )
                        )
                        return src_node, dest_node
        return None

    def distance(self, other: "Genome") -> float:
        """ Calculates the distance between two genomes.

        The shorter the distance between two genomes, the greater the similarity
        between them is. In the context of NEAT, the similarity between genomes
        increases as:

            1) the number of matching connection genes increases;
            2) the absolute difference between the matching connections weights
               decreases;

        The distance between genomes is used for speciation and for sexual
        reproduction (mating).

        The formula used is shown below. It's the same as the one presented in
        the original NEAT paper :cite:`stanley:ec02`. All the coefficients are
        configurable.

        .. math::
                \\delta = c_1 \\cdot \\frac{E}{N} + c_2 \\cdot \\frac{D}{N} \\
                    + c_3 \\cdot W
            :label: neat_genome_distance

        Args:
            other (NeatGenome): The other genome (an instance of
                :class:`.NeatGenome` or one of its subclasses).

        Returns:
            The distance between the genomes.
        """
        genes = align_connections(self.connections, other.connections)
        excess = disjoint = num_matches = 0
        weight_diff = 0.0

        g1_max_innov = np.amax([c.get_id() for c in self.connections])
        g2_max_innov = np.amax([c.get_id() for c in other.connections])

        for cn1, cn2 in zip(*genes):
            # non-matching genes:
            if cn1 is None or cn2 is None:
                # if c1 is None, c2 can't be None (and vice-versa)
                # noinspection PyUnresolvedReferences
                if (
                    (cn1 is None and cn2.get_id() > g1_max_innov)  # type: ignore
                    or (cn2 is None and cn1.get_id() > g2_max_innov)  # type: ignore
                ):
                    excess += 1
                else:
                    disjoint += 1
            # matching genes:
            else:
                num_matches += 1
                weight_diff += abs(cn1.weight - cn2.weight)

        c1 = self._config.excess_genes_coefficient
        c2 = self._config.disjoint_genes_coefficient
        c3 = self._config.weight_difference_coefficient

        n = max(len(self.connections), len(other.connections))
        return ((c1 * excess + c2 * disjoint) / n) + c3 * weight_diff / num_matches

    def copy_with_random_weights(self) -> "Genome":
        input_nodes = self.copy_nodes(self.input_nodes)
        output_nodes = self.copy_nodes(self.output_nodes)
        hidden_nodes = self.copy_nodes(self.hidden_nodes)
        bias_node = self.copy_nodes([self.bias_node])[0] if self.bias_node else None

        new_nodes = [] + input_nodes + output_nodes + hidden_nodes
        new_nodes += [bias_node] if bias_node is not None else []
        return Genome(
            input_nodes=input_nodes,
            hidden_nodes=hidden_nodes,
            output_nodes=output_nodes,
            bias_node=bias_node,
            config=self._config,
            connections=self.copy_connections(
                self.connections, new_nodes=new_nodes, with_random_weights=True
            ),
        )

    def clone(self) -> "Genome":
        input_nodes = self.copy_nodes(self.input_nodes)
        output_nodes = self.copy_nodes(self.output_nodes)
        hidden_nodes = self.copy_nodes(self.hidden_nodes)
        bias_node = self.copy_nodes([self.bias_node])[0] if self.bias_node else None

        new_nodes = [] + input_nodes + output_nodes + hidden_nodes
        new_nodes += [bias_node] if bias_node is not None else []

        return Genome(
            input_nodes=input_nodes,
            hidden_nodes=hidden_nodes,
            output_nodes=output_nodes,
            bias_node=bias_node,
            config=self._config,
            connections=self.copy_connections(
                self.connections, new_nodes=new_nodes, with_random_weights=False
            ),
        )

    def add_random_hidden_node(self, id_handler: IDHandler) -> NodeGene | None:
        eligible_connections = [conn for conn in self.connections if conn.is_enabled]

        if len(eligible_connections) == 0:
            return None

        np.random.shuffle(eligible_connections)  # type: ignore

        original_conn = eligible_connections[0]

        src_node = original_conn.get_source_node()
        dst_node = original_conn.get_destination_node()

        new_node = NodeGene(
            node_id=id_handler.get_next_hidden_node_id(
                src_node_id=src_node.get_id(), dst_node_id=dst_node.get_id()
            ),
            node_type=NodeGene.NodeTypeEnum.HIDDEN,
            activation_func=self._config.hidden_nodes_activation,
        )

        self.hidden_nodes.append(new_node)
        # disable old connection
        original_conn.is_enabled = False

        # add connections
        self.connections.append(
            self._create_connection(
                connection_id=id_handler.get_next_connection_id(
                    src_node_id=src_node.get_id(), dst_node_id=new_node.get_id()
                ),
                src_node=src_node,
                dst_node=new_node,
                weight=1,
            )
        )

        self.connections.append(
            self._create_connection(
                connection_id=id_handler.get_next_connection_id(
                    src_node_id=new_node.get_id(), dst_node_id=dst_node.get_id()
                ),
                src_node=new_node,
                dst_node=dst_node,
                weight=original_conn.weight,
            )
        )

        return new_node

    def valid_out_nodes(self) -> bool:
        """Checks if all the genome's output nodes are valid.

        An output node is considered to be valid if it receives, during its
        processing, at least one input, i.e., the node has at least one enabled
        incoming connection. Invalid output nodes simply outputs a fixed
        default value and are, in many cases, undesirable.

        Returns:
            `True` if all the genome's output nodes have at least one enabled
            incoming connection and `False` otherwise. Self-connecting
            connections are not considered.
        """
        for out_node in self.output_nodes:
            valid = False
            connections = [
                con
                for con in self.connections
                if con.get_destination_node() == out_node
            ]
            for in_con in connections:
                if in_con.is_enabled:
                    valid = True
                    break
            if not valid:
                return False
        return True

    def valid_in_nodes(self) -> bool:
        """Checks if all the genome's input nodes are valid.

        An input node is considered to be valid if it has at least one enabled
        connection leaving it, i.e., its activation is used as input by at least
        one other node.

        Returns:
            `True` if all the genome's input nodes are valid and `False`
            otherwise.
        """
        for in_node in self.input_nodes:
            valid = False
            connections = [
                con for con in self.connections if con.get_source_node() == in_node
            ]
            for out_con in connections:
                if out_con.is_enabled:
                    valid = True
                    break
            if not valid:
                return False
        return True

    @classmethod
    def _create_connection_with_random_weights(
        cls,
        connection_id: int,
        src_node: NodeGene,
        dst_node: NodeGene,
        config: NeatConfig,
    ) -> ConnectionGene:
        weight = np.random.uniform(*config.new_weight_interval)
        return cls._create_connection(
            connection_id, src_node, dst_node, weight, is_enabled=True
        )

    @classmethod
    def _create_connection(
        cls,
        connection_id: int,
        src_node: NodeGene,
        dst_node: NodeGene,
        weight: float,
        is_enabled: bool = True,
    ) -> ConnectionGene:
        if dst_node.get_type() in (
            NodeGene.NodeTypeEnum.BIAS,
            NodeGene.NodeTypeEnum.INPUT,
        ):
            raise ValueError(
                f"Attempt to create a connection pointing to a bias or input "
                f"node ({src_node.get_id()}->{dst_node.get_id()}). Nodes of this type "
                f"don't process input."
            )

        connection = ConnectionGene(
            connection_id=connection_id,
            source_node=src_node,
            destination_node=dst_node,
            weight=weight,
            is_enabled=is_enabled,
        )

        # src_node.add_output_connection(connection)
        # dst_node.add_input_connection(connection)
        return connection

    def copy_nodes(self, nodes: list[NodeGene]) -> list[NodeGene]:
        return [node.copy() for node in nodes]

    def copy_connections(
        self,
        connections: list[ConnectionGene],
        new_nodes: list[NodeGene],
        with_random_weights: bool,
    ) -> list[ConnectionGene]:
        new_connections: list[ConnectionGene] = []
        nodes_lookup = {node.get_id(): node for node in new_nodes}

        for connection in connections:
            if (
                nodes_lookup[connection.get_destination_node().get_id()].get_type()
                != connection.get_destination_node().get_type()
            ):
                print("here")
            if with_random_weights:
                new_conn = self._create_connection_with_random_weights(
                    connection_id=connection.get_id(),
                    src_node=nodes_lookup[connection.get_source_node().get_id()],
                    dst_node=nodes_lookup[connection.get_destination_node().get_id()],
                    config=self._config,
                )
            else:
                new_conn = self._create_connection(
                    connection_id=connection.get_id(),
                    src_node=nodes_lookup[connection.get_source_node().get_id()],
                    dst_node=nodes_lookup[connection.get_destination_node().get_id()],
                    weight=connection.weight,
                )

            new_connections.append(new_conn)

        return new_connections

    def does_connection_exists(self, src_node: NodeGene, dst_node: NodeGene) -> bool:
        # for con in src_node.get_connections_out():
        #     if con.get_destination_node().get_id() == dst_node.get_id():
        #         return True

        # return False

        for con in self.connections:
            if (
                con.get_source_node() == src_node
                and con.get_destination_node() == dst_node
            ):
                return True

        return False

    def get_node_connections_in(self, node: NodeGene) -> list[ConnectionGene]:
        return [con for con in self.connections if con.get_destination_node() == node]

    def get_node_connections_out(self, node: NodeGene) -> list[ConnectionGene]:
        return [con for con in self.connections if con.get_source_node() == node]

    # serialization

    def save(self, path: pathlib.Path) -> None:
        pickle_save(self, path.absolute())

    @classmethod
    def load(cls: type["Genome"], path: pathlib.Path) -> "Genome":
        obj = pickle_load(path.absolute())
        assert isinstance(
            obj, cls
        ), f"loaded object is not an instance of {cls.__name__}"
        return obj
