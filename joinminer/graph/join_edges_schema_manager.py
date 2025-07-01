import copy

def reverse_join_edges_schema(join_edges_schema):
    node_index_to_reversed_index = {}
    
    reversed_join_edges_schema = []
    for join_edge_i in range(len(join_edges_schema)):
        join_edge_schema = join_edges_schema[- (1 + join_edge_i)]
        
        # 先获得一个复制版本
        reversed_join_edge_schema = copy.deepcopy(join_edge_schema)
        
        # 颠倒join_nodes
        reversed_join_edge_schema["join_nodes_types"] = list(join_edge_schema["add_nodes_types"])
        reversed_join_edge_schema["join_nodes_columns"] = list(join_edge_schema["add_nodes_columns"])
        reversed_join_edge_schema["join_nodes_indexes"] = list(join_edge_schema["add_nodes_indexes"])
        
        # 颠倒add_nodes
        reversed_join_edge_schema["add_nodes_types"] = list(join_edge_schema["join_nodes_types"])
        reversed_join_edge_schema["add_nodes_columns"] = list(join_edge_schema["join_nodes_columns"])
        reversed_join_edge_schema["add_nodes_indexes"] = list(join_edge_schema["join_nodes_indexes"])
        
        reversed_join_edges_schema.append(reversed_join_edge_schema)
        
    return reversed_join_edges_schema