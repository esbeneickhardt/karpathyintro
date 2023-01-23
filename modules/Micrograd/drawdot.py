from graphviz import Digraph
from value import Value

def trace(root: Value) -> tuple:
    """
    Description:
        Iterates recursively through the previous calculations
        from the root to the leafs and creates a tree.
    Inputs:
        root: The Value object of a forward pass
    Outputs:
        Nodes and Edges representing the calculations
    """
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root: Value) -> Digraph:
    """
    Description:
        Creates a graph showing the relations between calculations.
    Inputs:
        root: The Value object of a forward pass
    Outputs:
        A graph representation of the calculations        
    """
    nodes, edges = trace(root)
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    for n in nodes:
        uid = str(id(n))
        # For each value in graph create rectangular node for it
        dot.node(name=uid, label = "{ %s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape='record')
        
        # If the value is a result of a calculation create a "operation" node for it
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        # Connect the n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot