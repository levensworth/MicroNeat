

class IDHandler:
    def __init__(self,
                 num_inputs: int,
                 num_outputs: int,
                 has_bias: bool) -> None:
        self._node_counter = num_inputs + num_outputs + (1 if has_bias else 0)
        self._connection_counter = num_inputs * num_outputs
        self._species_counter = 0
        self._new_connections_ids: dict[int, dict[int, int]] = {}
        self._new_nodes_ids: dict[int, dict[int, int]] = {}
        self.reset_counter = 0

    def next_species_id(self) -> int:
        """ Returns a new unique ID for a species. """
        sid = self._species_counter
        self._species_counter += 1
        return sid
    
    def get_next_hidden_node_id(self, src_node_id: int, dst_node_id: int) -> int:
        if src_node_id is None or dst_node_id is None:
            raise RuntimeError("Trying to generate an ID to a node whose "
                               "parents (one or both) have \"None\" IDs!")

        if src_node_id in self._new_nodes_ids:
            if dst_node_id in self._new_nodes_ids[src_node_id]:
                return self._new_nodes_ids[src_node_id][dst_node_id]
        else:
            self._new_nodes_ids[src_node_id] = {}

        hid = self._node_counter
        self._node_counter += 1
        self._new_nodes_ids[src_node_id][dst_node_id] = hid
        return hid
    
    def get_next_connection_id(self, src_node_id: int, dst_node_id: int) -> int:
        if src_node_id in self._new_connections_ids:
            if dst_node_id in self._new_connections_ids[src_node_id]:
                return self._new_connections_ids[src_node_id][dst_node_id]
        else:
            self._new_connections_ids[src_node_id] = {}

        cid = self._connection_counter
        self._connection_counter += 1
        self._new_connections_ids[src_node_id][dst_node_id] = cid
        return cid