import enum
import typing


class NodeGene:
    def __init__(self, node_id: int, node_type: "NodeGene.NodeTypeEnum", activation_func: typing.Callable[[float], float]) -> None:
        self._id = node_id
        self._type = node_type
        self._activation_func = activation_func
        
        self._connections_in: list['ConnectionGene'] = []
        self._connections_out: list['ConnectionGene'] = []

    class NodeTypeEnum(str, enum.Enum):
        INPUT = 'INPUT'
        OUTPUT = 'OUTPUT'
        HIDDEN = 'HIDDEN'
        BIAS = 'BIAS'

    def __hash__(self) -> int:
        return hash(self._id)

    def get_id(self) -> int:
        return self._id
    
    def get_type(self) -> "NodeGene.NodeTypeEnum":
        return self._type
    
    def activate(self, value: float) -> float:
        if self._type == NodeGene.NodeTypeEnum.BIAS:
            pass
        return self._activation_func(value)
    
    def copy(self,) -> "NodeGene":
        """Returns a shallow copy of the node, without it's connections.

        Returns:
            NodeGene: new instance of the node without connections.
        """
        new_node = NodeGene(
            node_id=self.get_id(),
            node_type=self._type,
            activation_func=self._activation_func,
        )

        return new_node
    
    def __repr__(self) -> str:
        return f'Node(id={self.get_id()} | type={self.get_type()}'
    

class ConnectionGene:
    def __init__(self, connection_id: int, source_node: NodeGene, destination_node: NodeGene, weight: float, is_enabled: bool = True) -> None:
        assert destination_node.get_type() != NodeGene.NodeTypeEnum.BIAS, 'you can not create a in connection to a bias node'
        self._id = connection_id
        self._src_node = source_node
        self._dst_node = destination_node
        self.weight = weight
        self.is_enabled = is_enabled

    def get_id(self) -> int:
        return self._id
    
    def get_source_node(self) -> NodeGene:
        return self._src_node
    
    def get_destination_node(self) -> NodeGene:
        return self._dst_node
    
    def is_self_connection(self) -> bool:
        return self._src_node.get_id() == self._dst_node.get_id()
    
    def __repr__(self) -> str:
        return f'Con [{self.get_id()}][{self.is_enabled}] {self.get_source_node().get_id()} -> {self.get_destination_node().get_id()}'
    


def align_connections(
    con_list1: list[ConnectionGene], 
    con_list2: list[ConnectionGene], 
    is_debug: bool = False
) -> tuple[list[ConnectionGene | None], list[ConnectionGene | None]]:
    con_dict1 = {c.get_id(): c for c in con_list1}
    con_dict2 = {c.get_id(): c for c in con_list2}
    union = sorted(set(con_dict1.keys()) | set(con_dict2.keys()))

    aligned1: list[ConnectionGene | None] = []
    aligned2: list[ConnectionGene | None] = []
    for cid in union:
        aligned1.append(con_dict1[cid] if cid in con_dict1 else None)
        aligned2.append(con_dict2[cid] if cid in con_dict2 else None)

    # debug
    if is_debug:
        for c1, c2 in zip(aligned1, aligned2):
            print(c1.get_id() if c1 is not None else "-", end=" | ")
            print(c2.get_id() if c2 is not None else "-")

    return aligned1, aligned2