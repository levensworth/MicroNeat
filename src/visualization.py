import random
import plotly.graph_objects as go # type: ignore

from src.genes import NodeGene, ConnectionGene
from src.genome import Genome
from dataclasses import dataclass

@dataclass
class GraphNode:
    x_pos: float
    y_pos: float
    id: int

@dataclass
class GraphEdge:
    x_src: float
    y_src: float

    x_dst: float
    y_dst: float

    value: float


def visualize_genome(genome: Genome) -> None:
    un_mapped_layers = map_to_layers(genome)
    max_width = max([len(layer) for layer in un_mapped_layers])
    layers = [map_layer_to_graph(layer, layer_idx, max_width) for layer_idx, layer in enumerate(un_mapped_layers)]
    nodes: list[GraphNode] = []
    for layer in layers:
        nodes += layer
    connections = map_connections(connections=genome.connections, nodes=nodes)
    # Create edges for the DAG
    edge_trace = []
    for edge in connections:
        edge_trace.append(go.Scatter(
            x=[edge.x_src, edge.x_dst, None],
            y=[edge.y_src, edge.y_dst, None],
            mode='lines',
            line=dict(width=abs(2*edge.value), color='blue'),
            text=edge.value,
            hovertext='text'
        ))

    # Create nodes for the DAG
    node_trace = go.Scatter(
        x=[node.x_pos for node in nodes],
        y=[node.y_pos for node in nodes],
        mode='markers+text',
        text=[node.id for node in nodes],
        textposition='top center',
        hovertext=[f'{node.id}<br>{(node.x_pos, node.y_pos)}' for node in nodes],
        marker=dict(
            size=10,
            color='red',
            line_width=2
        )
    )

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='DAG Visualization',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                    ))

    fig.show()

            

def map_to_layers(genome: Genome) -> list[list[NodeGene]]:
    """Algorithm:
    1. init layers with input nodes
    2. create a queue and put all nodes connected to the inputs, associated with level 1
    3. for each node, if all inputs to that node are already mapped to a layer 
        or are included in the queue => then map the current level
    4. if prev step yield that at least 1 in_edge is not present, 
        * add node to the queue associated with level lvl
        * add current node to the queue in lvl + 1
    


    Args:
        genome (Genome): _description_

    Returns:
        list[list[NodeGene]]: _description_
    """
    layers: list[list[NodeGene]] = []
    layers.append(genome.input_nodes)
    
    visited: set[NodeGene] = set(genome.input_nodes)    
    queue: list[tuple[NodeGene, int]] = []
    queue_lookup = {}
    # init queue with direct links of inputs
    
    for con in [con for con in genome.connections if con.get_source_node() in visited]:
        if not con.is_enabled:
            continue
        new_node = con.get_destination_node() 
        if new_node in visited:
            continue

        if new_node.get_type() == NodeGene.NodeTypeEnum.OUTPUT:
            continue
        
        queue.append((new_node, 1))
        queue_lookup[new_node] = 1

    queue =  list(set(queue)) # remove duplicate entries
    

    assigned: set[tuple[int, NodeGene]] = set()
    nodes_assigned: set[NodeGene] = set()
    while len(queue) > 0:
        current, lvl = queue.pop(0)
        if current in nodes_assigned:
            continue

        lvl = queue_lookup[current]
        # add outputs to queue
        for con in current.get_connections_out():
            if con.get_destination_node().get_type() == NodeGene.NodeTypeEnum.OUTPUT:
                continue

            if con.get_destination_node() in visited:
                continue
            queue.append((con.get_destination_node(), lvl + 1))
            queue_lookup.setdefault(con.get_destination_node(), lvl + 1)
            queue_lookup[con.get_destination_node()] = max(queue_lookup[con.get_destination_node()], lvl + 1)

        # check fon unseen inputs
        has_unseen_inputs = False
        for con in current.get_connections_in():
            if con.get_source_node() in visited:
                continue

            # if connection is from an output, don't count it
            if con.get_source_node().get_type() == NodeGene.NodeTypeEnum.OUTPUT:
                continue
            
            in_queue = [t[0] == con.get_source_node() for t in queue if t[1] <= lvl]
            if any(in_queue):
                continue

            else:
                queue.append((con.get_source_node(), lvl))
                queue_lookup[con.get_source_node()] = lvl
            
            
            has_unseen_inputs = True
            break
        
        if has_unseen_inputs:
            # there are some nodes which are inputs to me and they are yet to be included
            queue.append((current, lvl + 1))
            queue_lookup[current] = max(queue_lookup[current], lvl + 1)
            continue

        # all of my inputs are already in placed or in queue
        assigned.add((lvl, current))
        nodes_assigned.add(current)
        visited.add(current)

                
        
        
    current_lvl = 1
    current_layer: list[NodeGene] = []
    aux = list(assigned)
    aux.sort(key=lambda t: t[0])
    for lvl,node in  aux:
        if lvl > current_lvl:
            layers.append(current_layer)
            current_layer = []
            current_lvl = lvl
        
        current_layer.append(node)
    if current_layer != []:
        layers.append(current_layer)
    
    # if genome.bias_node is not None:
    #     layers.append([genome.bias_node])
    layers.append(genome.output_nodes)
    return layers


def map_layer_to_graph(layer: list[NodeGene], layer_number: int, max_width: int) -> list[GraphNode]:
    graph_nodes: list[GraphNode] = []
    y_delta = 2
    x_delta = max_width//len(layer)
    for idx, node in enumerate(layer):
        graph_nodes.append(GraphNode(
            x_pos=x_delta * idx + random.random() * 1,
            y_pos=y_delta * layer_number,
            id=node.get_id()
        ))
    
    return graph_nodes

def map_connections(connections: list[ConnectionGene], nodes: list[GraphNode]) -> list[GraphEdge]:
    conns = []
    node_lookup_table = {node.id: node for node in nodes}
    for con in connections:
        if not con.is_enabled:
            continue
        
        if con.get_destination_node().get_id() not in node_lookup_table:
            print(f'connection {con} was skipped because the destination does not exists')
            continue
        if con.get_source_node().get_id() not in node_lookup_table:
            print(f'connection {con} was skipped because the source does not exists')
            continue

        conns.append(GraphEdge(
            x_src=node_lookup_table[con.get_source_node().get_id()].x_pos,
            y_src=node_lookup_table[con.get_source_node().get_id()].y_pos,
            x_dst=node_lookup_table[con.get_destination_node().get_id()].x_pos,
            y_dst=node_lookup_table[con.get_destination_node().get_id()].y_pos,
            value=con.weight
        ))    

    return conns


def has_connections_to(node: NodeGene, dst_nodes: list[NodeGene]) -> bool:
    node_inputs = set([con.get_source_node().get_id() for con in node.get_connections_in()])
    for node in dst_nodes:
        if node.get_id() in node_inputs:
            return True
    
    return False
    
    
        
        