import json
import re
import requests

from pretrain.schema import DATATYPE, TYPE_OF, CONCEPT, RELATED_CONCEPT
from reflect.sr_statement import SRFORStatement, SRWhileStatement


class MethodKG:
    def __init__(self, code, language, sr_method=None):
        self.code = code
        self.language = language
        self.sr_method = sr_method
        self.language_keywords_map = {
            "java": "override class print java String false synchronized int abstract float private char boolean var static null if const for true while long strictfp finally protected import native final void enum else break transient catch instanceof byte super volatile case assert short package default double public try this switch continue throws protected public private module requires exports do return",
            "javascript":"override class print javascript in of if for while finally var new function do return void else break catch instanceof with throw case default try this switch continue typeof delete let yield const export super debugger as async await static import from as",
            "python":"override class print python and elif is global as in if from raise for except finally print import pass return exec else break not with class assert yield try while continue del or def lambda async await nonlocal|10",
            "cpp":"override class print c++ if else elif endif define undef warning error line pragma _Pragma ifdef ifndef include int float while private char char8_t char16_t char32_t catch import module export virtual operator sizeof dynamic_cast|10 typedef const_cast|10 const for static_cast|10 union namespace unsigned long volatile static protected bool template mutable if public friend do goto auto void enum else break extern using asm case typeid wchar_tshort reinterpret_cast|10 default double register explicit signed typename try this switch continue inline delete alignas alignof constexpr consteval constinit decltype concept co_await co_return co_yield requires noexcept static_assert thread_local restrict final override atomic_bool atomic_char atomic_schar atomic_uchar atomic_short atomic_ushort atomic_int atomic_uint atomic_long atomic_ulong atomic_llong atomic_ullong new throw return and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq",
            "c-sharp":"override class print c# abstract as base bool break byte case catch char checked const continue decimal default delegate do double enum event explicit extern finally fixed float for foreach goto if implicit in int interface internal is lock long object operator out override params private protected public readonly ref sbyte sealed short sizeof stackalloc static string struct switch this try typeof uint ulong unchecked unsafe ushort using virtual void volatile while add alias ascending async await by descending dynamic equals from get global group into join let nameof on orderby partial remove select set value var when where yield",
            "c": "override class print c if else elif endif define undef warning error line pragma _Pragma ifdef ifndef include int float while private char char8_t char16_t char32_t catch import module export virtual operator sizeof dynamic_cast|10 typedef const_cast|10 const for static_cast|10 union namespace unsigned long volatile static protected bool template mutable if public friend do goto auto void enum else break extern using asm case typeid wchar_tshort reinterpret_cast|10 default double register explicit signed typename try this switch continue inline delete alignas alignof constexpr consteval constinit decltype concept co_await co_return co_yield requires noexcept static_assert thread_local restrict final override atomic_bool atomic_char atomic_schar atomic_uchar atomic_short atomic_ushort atomic_int atomic_uint atomic_long atomic_ulong atomic_llong atomic_ullong new throw return and and_eq bitand bitor compl not not_eq or or_eq xor xor_eq",
            "go":"override class print go break default func interface select case map struct chan else goto package switch const fallthrough if range type continue for import return var go defer bool byte complex64 complex128 float32 float64 int8 int16 int32 int64 string uint8 uint16 uint32 uint64 int uint uintptr rune",
            "ruby":"override class print ruby and then defined module in return redo if BEGIN retry end for self when next until do begin unless END rescue else break undef not super class case require yield alias while ensure elsif or include attr_reader attr_writer attr_accessor",
            "dart":"dispose Key required Curves BuildContext duration late State initState createState build Widget StatefulWidget StatelessWidget BuildContext return override class print dart int abstract as assert async await break case catch class const continue covariant default deferred do dynamic else enum export extends extension external factory false final finally for Function get hide if implements import in inferface is library mixin new null on operator part rethrow return set show static super switch sync this throw true try typedef var void while with yield",
        }
        self.language_symbols=[
            "+", "-", "*", "/", "%", "++", "--",  # Arithmetic operators
            "=", "+=", "-=", "*=", "/=", "%=",    # Assignment operators
            "==", "!=", ">", "<", ">=", "<=",     # Comparison operators
            "&&", "||", "!",                      # Logical operators
            "&", "|", "^", "~", "<<", ">>",       # Bitwise operators
            ";", ",", ".", ":", "::",             # Punctuation/Delimiters
            "()", "{}", "[]", "<>",               # Parentheses and Brackets
            "? :", "=>", "->", ".", "@", "#",     # Special operators
            "\"", "'",                            # String and Character symbols
            "\\n", "\\t", "\\\\", "\\'", "\\\"",  # Escape sequences
            "$", "&", "*", "/* */", "//"          # Miscellaneous symbols
        ]
        self.nodes = []
        self.edges = []
        self.language_keywords = self.language_keywords_map[self.language].split(" ")

    def contains_digit(self, s):
        if s in self.language_symbols:
            return True
        else:
            return False

    def parse_tokens(self):
        code_lines = self.code.split("\n")
        for line in code_lines:
            if line.startswith("#") or line.startswith("//") or line.startswith("import"):
                continue

            if "=" in line and "==" not in line:
                line_split = line.split("=")
                pre_line = line_split[0]
                tail_line = line_split[1]
                pre_line_token_l = self.tokenize_code(pre_line)

                if len(pre_line_token_l) == 2:
                    predict_var = pre_line_token_l[1]
                    predict_identifier = pre_line_token_l[0]
                    if predict_var not in self.language_keywords and predict_identifier in self.language_keywords:
                        predict_var_node = self.get_or_create_node(name=predict_var, label=IDENTIFIER)
                        predict_identifier = self.get_or_create_node(name=predict_identifier, label=DATATYPE)
                        predict_type_of_edge = self.get_or_create_edge(label=TYPE_OF, source=predict_identifier.id, target=predict_var_node.id)

                        # self.type_of_edge.append([predict_var, predict_identifier])
                        # self.edges.append(Edge())
                        #
                        # self.tokens.append(predict_var)
                        # self.identifiers.append(predict_identifier)
            line_token_l = self.tokenize_code(line)
            for token in line_token_l:

                if self.contains_digit(token):
                    continue

                if self.is_preposition(token):
                    continue

                if len(token) <= 1:
                    continue

                if self.is_article_word(token):
                    continue

                # language = langid.classify(token)[0]
                # if language != "en":
                #     continue

                if token in self.language_keywords:
                    self.get_or_create_node(name=token, label=DATATYPE)
                else:
                    self.get_or_create_node(name=token, label=IDENTIFIER)

        # print(self.tokens)

    def fetch_identifier(self, token):
        if self.contains_digit(token):
            return None

        if self.is_preposition(token):
            return None

        if len(token) <= 1:
            return None

        if self.is_article_word(token):
            return None

        if token in self.language_keywords:
            return self.get_or_create_node(name=token, label=DATATYPE)
        else:
            return self.get_or_create_node(name=token, label=IDENTIFIER)


    def parse_control_dependence(self):
        for statement in self.sr_method.statement_list:
            if type(statement) is SRFORStatement or type(statement) is SRWhileStatement:
                for_identifiers = []
                for token in statement.local_word_list:
                    fetch_result = self.fetch_identifier(token)
                    if fetch_result is not None:
                        if fetch_result.label == IDENTIFIER:
                            for_identifiers.append(token)
                print(for_identifiers)

    def is_preposition(self, word):
        prepositions = {
            "about", "above", "across", "after", "against", "along", "among",
            "around", "at", "before", "behind", "below", "beneath", "beside",
            "between", "beyond", "by", "down", "during", "except", "for",
            "from", "in", "inside", "into", "like", "near", "of", "off",
            "on", "out", "outside", "over", "past", "since", "through",
            "throughout", "to", "toward", "under", "until", "up", "upon",
            "with", "within", "without"
        }
        return word.lower() in prepositions

    def is_article_word(self, word):
        articles = {"the", "a", "an"}
        return word.lower() in articles

    def tokenize_code(self, code):
        # 正则表达式，只匹配字母、数字和下划线（标识符、数字）
        token_pattern = r'\b\w+\b'

        # 使用 re.findall 来获取所有匹配的 token
        tokens = re.findall(token_pattern, code)

        return tokens

    def parse_concept_nodes(self):
        for node in self.nodes:
            if node.label == IDENTIFIER:
                concepts = self.parse_concepts(node.name)
                for concept in concepts:
                    concept_node = self.get_or_create_node(name=concept, label=CONCEPT)
                    related_concept_edge = self.get_or_create_edge(label=RELATED_CONCEPT, source=node.id, target=concept_node.id)
    def parse_concepts(self, token):
        words = self.split_variable_name(token)
        return words

    def split_variable_name(self, name):
        if '_' in name:
            # 处理 snake_case
            return name.split('_')
        else:
            # 处理 CamelCase
            return re.sub('([a-z])([A-Z])', r'\1 \2', name).split()

    def get_or_create_node(self, name, label):
        for node in self.nodes:
            if node.name == name and node.label == label:
                return node
        new_node = Node(name, label, len(self.nodes))
        self.nodes.append(new_node)
        return new_node

    def get_or_create_edge(self, label, source, target):
        for edge in self.edges:
            if edge.label == label and edge.source == source and edge.target == target:
                return edge

        new_edge = Edge(label, source, target, len(self.edges))
        self.edges.append(new_edge)
        return new_edge

class Node:
    def __init__(self, name, label, id):
        self.name = name
        self.label = label
        self.id = id


class Edge:
    def __init__(self, label, source, target, id):
        self.label = label
        self.source = source
        self.target = target
        self.id = id