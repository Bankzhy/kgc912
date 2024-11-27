import re
import pymysql
from pretrain.schema import VAR_IDENTIFIER, METHOD_IDENTIFIER, CONCEPT, RELATED_CONCEPT, VAR_ASSIGNMENT, DATA_DEPENDENCY


class MKG:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.schema_edges = [
            'type_of',
            "control_dependency",
            "data_dependency",
            "has_method",
            "has_property",
            "assignment",
            "related_concept"
        ]
        # Connect to the MySQL database
        self.conn = pymysql.connect(
            host="47.113.220.80",
            user="root",
            password="Apple3328823%",
            database="kgc",
            charset="utf8mb4",  # Use utf8mb4 for full Unicode support,
            connect_timeout=50
        )
        self.cursor = self.conn.cursor()
        self.max_node_expand_num = 10

    def get_or_create_node(self, name, type):
        for node in self.nodes:
            if node.label == name and node.type == type:
                return node, False
        new_node = Node(name, type)
        self.nodes.append(new_node)
        return new_node, True

    def get_node(self, name, type):
        for node in self.nodes:
            if node.label == name and node.type == type:
                return node
        return None


    def check_assignment_var(self, name, node_label):
        var_name = "".join(node_label.rsplit("_", 1)[0])
        if var_name == name:
            return True
        else:
            return False

    def get_max_assignment_var_node(self, name):
        max_num = 0
        max_node = None
        for node in self.nodes:
            if node.type == VAR_ASSIGNMENT:
                if self.check_assignment_var(name, node.label):
                    naml = node.label.split("_")
                    if len(naml) > 1:
                        index = int(naml[len(naml)-1])
                        if index >= max_num:
                            max_num = index
                            max_node = node
                    else:
                        return None, 0
        return max_node,max_num

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

    def parse_method_name(self, method_name):
        new_method, created = self.get_or_create_node(method_name, METHOD_IDENTIFIER)
        method_name_l = self.split_variable_name(method_name)
        for token in method_name_l:
            new_token, created = self.get_or_create_node(token, CONCEPT)
            new_edge = self.get_or_create_edge(new_method, new_token, RELATED_CONCEPT)

    def parse_concept(self):
        for node in self.nodes:
            if node.type == VAR_IDENTIFIER or node.type == METHOD_IDENTIFIER:
                concept_l = self.split_variable_name(node.label)
                if len(concept_l) > 1:
                    for concept in concept_l:
                        lower_concept = concept.lower()
                        new_concept_node, created = self.get_or_create_node(lower_concept, CONCEPT)
                        new_concept_edge = self.get_or_create_edge(node, new_concept_node, RELATED_CONCEPT)
    def expand_concept_edge(self):
        concept_l = []
        for node in self.nodes:
            if node.type == CONCEPT:
                concept_l.append(node)
        if len(concept_l) > 2:
            for i in range(len(concept_l)):
                for j in range(i+1, len(concept_l)):
                    row_0 = self.fetch_concept(concept_l[i].label, concept_l[j].label)
                    row_1 = self.fetch_concept(concept_l[j].label, concept_l[i].label)
                    if row_0 is None and row_1 is None:
                        continue

                    rel_0 = row_0[3]
                    if rel_0 is not None:
                        self.get_or_create_edge(concept_l[i], concept_l[j], rel_0)

                    rel_1 = row_1[3]
                    if rel_1 is not None:
                        self.get_or_create_edge(concept_l[j], concept_l[i], rel_1)

    def expand_concept_node(self, method_name):
        method_name_l = self.split_variable_name(method_name)
        for token in method_name_l:
            token_node = self.get_node(token, CONCEPT)
            concept_nodes = self.fetch_concept_node(token)
            if concept_nodes is None:
                return
            print("expand node count: ", len(concept_nodes))
            print(concept_nodes)
            for cn in concept_nodes:
                if cn[1] == token:
                    new_concept_node, created = self.get_or_create_node(cn[2], CONCEPT)
                    new_concept_edge = self.get_or_create_edge(token_node, new_concept_node, cn[3])

                if cn[2] == token:
                    new_concept_node, created = self.get_or_create_node(cn[1], CONCEPT)
                    new_concept_edge = self.get_or_create_edge(token_node, new_concept_node, cn[3])

    def fetch_concept_node(self, concept):
        query = "SELECT * FROM conceptnet5 WHERE (arg1 = %s or arg2 = %s) and rel != 'DerivedFrom'"
        self.cursor.execute(query, (concept, concept))
        row = self.cursor.fetchall()  # Fetch a single row

        if row:
            return row  # Return the row if it exists
        else:
            return None  # Return None if no row is found

    def fetch_concept(self, concept1, concept2):
        query = "SELECT * FROM conceptnet5 WHERE arg1 = %s AND arg2 = %s"
        self.cursor.execute(query, (concept1, concept2))
        row = self.cursor.fetchone()  # Fetch a single row

        if row:
            return row  # Return the row if it exists
        else:
            return None  # Return None if no row is found

    def get_start_assignment_node(self, end_node):
        for edge in self.edges:
            if edge.type == DATA_DEPENDENCY:
                if edge.target.label == end_node.label:
                    return edge.source

    def find_edge(self, node1, node2):
        for edge in self.edges:
            if edge.source.label == node1 and edge.target.label == node2:
                return edge
        return None

    def to_dict(self):
        result = []
        for edge in self.edges:
            source = {
                "label": edge.source.label,
                "type": edge.source.type,
            }
            target = {
                "label": edge.target.label,
                "type": edge.target.type,
            }
            edge = {
                "source": source,
                "target": target,
                "type": edge.type,
            }
            result.append(edge)
        return result

class Node:
    def __init__(self, label, type):
        self.label = label
        self.type = type


class Edge:
    def __init__(self, source, target, type):
        self.type = type
        self.source = source
        self.target = target
