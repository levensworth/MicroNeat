import typing

if typing.TYPE_CHECKING:
    from src.genes import NodeGene
    from src.genome import Genome


class NeuralNetwork:
    
    def __init__(self, genome: "Genome") -> None:
        self._genome = genome
        self._activation_lookup: dict[int, float] = {}
        self._counter = 0

    def process(self, inputs: list[float]) -> list[float]:
        
        if not len(inputs) == self._genome.get_input_shape():
            raise ValueError('mismatched between the input size and the network')
        result = []
        
        for value, node in zip(inputs, self._genome.input_nodes):
            self._activation_lookup[node.get_id()] = node.activate(value)
        
        if self._genome.bias_node is not None:
            self._activation_lookup[self._genome.bias_node.get_id()] = self._genome.bias_node.activate(1)

        for node in self._genome.output_nodes:
            try:
                result.append(
                    self._apply_computation(node)
                )
            except Exception as e:
                raise e
        return result


    def _apply_computation(self, current_node: "NodeGene") -> float:
        current_value = 0.0
        self._activation_lookup[current_node.get_id()] = 0 # this is a measure to stop looping forever
        for con in self._genome.get_node_connections_in(current_node):
            if not con.is_enabled:
                continue
            if con.get_source_node().get_id() in self._activation_lookup:
                computation_value =  self._activation_lookup[con.get_source_node().get_id()]
            else:
                computation_value = self._apply_computation(con.get_source_node())
            
            current_value += con.weight * computation_value

        activated_value = current_node.activate(current_value)
        self._activation_lookup[current_node.get_id()] = activated_value
        return activated_value

        

        