import re

from pretrain.schema import VAR_IDENTIFIER, METHOD_IDENTIFIER, CONCEPT, RELATED_CONCEPT


class MKG:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def get_or_create_node(self, name, type):
        for node in self.nodes:
            if node.label == name and node.type == type:
                return node, False
        new_node = Node(name, type)
        self.nodes.append(new_node)
        return new_node, True

    def get_or_create_edge(self, source, target, type):
        for edge in self.edges:
            if edge.source == source and edge.target == target and edge.type == type:
                return edge
        new_edge = Edge(source, target, type)
        self.edges.append(new_edge)
        return new_edge

    def split_variable_name(self, name):
        if '_' in name:
            # 处理 snake_case
            return name.split('_')
        else:
            # 处理 CamelCase
            return re.sub('([a-z])([A-Z])', r'\1 \2', name).split()

    def parse_concept(self):
        for node in self.nodes:
            if node.type == VAR_IDENTIFIER or node.type == METHOD_IDENTIFIER:
                concept_l = self.split_variable_name(node.label)
                if len(concept_l) > 1:
                    for concept in concept_l:
                        lower_concept = concept.lower()
                        new_concept_node = self.get_or_create_node(lower_concept, CONCEPT)
                        new_concept_edge = self.get_or_create_edge(node, new_concept_node, RELATED_CONCEPT)
                print(concept_l)

class Node:
    def __init__(self, label, type):
        self.label = label
        self.type = type


class Edge:
    def __init__(self, source, target, type):
        self.type = type
        self.source = source
        self.target = target
