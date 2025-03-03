import os
import json
import pickle
import random
import logging
import re

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from vocab import Vocab
from data_utils import remove_comments_and_docstrings, replace_string_literal, tokenize_source

# from pretrain.vocab import Vocab
# from pretrain.data_utils import remove_comments_and_docstrings, replace_string_literal, tokenize_source

# from pretrain.vocab.vocab import Vocab

logger = logging.getLogger(__name__)


class KGCodeDataset(Dataset):
    spliter = " "
    KG_SEP_TOKEN = " "+ Vocab.KG_SEP_TOKEN + " "
    
    def __init__(self, args, task, split=None):
        self.args = args
        self.task = task
        self.dataset_dir = os.path.join(args.dataset_root)
        self.split = split
        self.codes = []
        self.structures = []
        self.nls = []
        self.docs = []

        self.codes_1 = []
        self.codes_2 = []
        self.labels = []

        self.st_type = [
            'type_of',
            "control_dependency",
            "data_dependency",
            "has_method",
            "has_property",
            "assignment"
        ]
        self.dataset_name = "KGCode"
        self.codes, self.structures, self.nls, self.docs = self.load_dataset_from_dir(dataset_dir=self.dataset_dir)

        self.size = len(self.codes)

    def split_edge_name(self, name):
            # 处理 CamelCase
        return re.sub('([a-z])([A-Z])', r'\1 \2', name).split()

    def __len__(self):
        return len(self.codes)

    def subset(self, ratio):
        """
        Return a subset of self.

        Args:
            ratio (float): The ratio of size, must greater than 0 and less than/equal to 1

        Returns:
            Dataset: the subset

        """
        assert 0 < ratio <= 1, f'The subset ratio supposed to be 0 < ratio <= 1, but got ratio={ratio}'
        if ratio == 1:
            return self
        indices = random.sample(range(self.size), int(self.size * ratio))
        return torch.utils.data.Subset(self, indices)

    def __getitem__(self, index):
        if self.task == "mass":
            code_tokens = self.codes[index].split()
            mask_len = int(self.args.mass_mask_ratio * len(code_tokens))
            mask_start = random.randint(0, len(code_tokens) - mask_len)
            mask_tokens = code_tokens[mask_start: mask_start + mask_len]
            input_tokens = code_tokens[:mask_start] + [Vocab.MSK_TOKEN] + code_tokens[mask_start + mask_len:]
            return ' '.join(input_tokens), self.structures[index], self.nls[index], ' '.join(mask_tokens)
        elif self.task == "rlp":
             label_l = []
             new_st_l = []
             structure = self.structures[index]
             st_l = structure.split(self.KG_SEP_TOKEN)
             for st in st_l:
                child_l = st.split(self.spliter)
                if len(child_l) < 3:
                    continue
                label_l.append(child_l[1])
                new_st = child_l[0] + self.spliter + Vocab.MSK_TOKEN + self.spliter + child_l[2]
                new_st_l.append(new_st)
             mask_st = self.KG_SEP_TOKEN.join(new_st_l)
             label = ",".join(label_l)
             return self.codes[index], mask_st, self.nls[index], label
        elif self.task == "nlp":
            return self.codes[index], self.structures[index], self.nls[index], self.docs[index]
        elif self.task == "cgp":
            structure = self.structures[index]
            st_l = structure.split(self.KG_SEP_TOKEN)
            is_correct = random.random() < 0.5
            if structure == "":
                is_correct = False

            if is_correct:
                target_st = random.choices(st_l, k=1)[0]
                st_l.remove(target_st)
                new_st = self.KG_SEP_TOKEN.join(st_l)
                return self.codes[index], new_st, self.nls[index], target_st, 1
            else:
                other_graph = self.structures[random.randint(0, len(self.structures) - 1)]
                while other_graph == self.structures[index]:
                    other_graph = self.structures[random.randint(0, len(self.structures) - 1)]
                other_stl = other_graph.split(self.KG_SEP_TOKEN)
                target_st = random.choices(other_stl, k=1)[0]
                return self.codes[index], self.structures[index], self.nls[index], target_st, 0


            # if self.structures[index]=="":
            #     return self.codes[index], self.structures[index], self.nls[index], 1
            #
            # is_graph = random.random() < 0.5
            # if is_graph:
            #     return self.codes[index], self.structures[index], self.nls[index], 1
            # else:
            #     other_graph = self.structures[random.randint(0, len(self.structures) - 1)]
            #     while other_graph == self.structures[index]:
            #         other_graph = self.structures[random.randint(0, len(self.structures) - 1)]
            #     return self.codes[index], other_graph, self.nls[index], 0

        elif self.task == "ngp":
            concept = self.nls[index]
            nls_l = concept.split(",")
            is_correct = random.random() < 0.5


        elif self.task == "rrlp":
            label_l = []
            new_st_l = []
            structure = self.structures[index]

            if structure == "":
                return self.codes[index], structure, self.nls[index], structure
            st_l = structure.split(self.KG_SEP_TOKEN)
            for st in st_l:
                child_l = st.split(self.spliter)
                if len(child_l) != 3:
                    continue

                if len(st_l) == 1:
                    new_st = child_l[0] + self.spliter + Vocab.MSK_TOKEN + self.spliter + child_l[2]
                    label_l.append(child_l[1])
                    new_st_l.append(new_st)
                    continue

                random_number = random.random()
                if random_number < 0.5 or len(label_l) <= 0:
                    new_st = child_l[0] + self.spliter + Vocab.MSK_TOKEN + self.spliter + child_l[2]
                    label_l.append(child_l[1])
                else:
                    new_st = child_l[0] + self.spliter + child_l[1] + self.spliter + child_l[2]
                new_st_l.append(new_st)
            mask_st = self.KG_SEP_TOKEN.join(new_st_l)
            label = ",".join(label_l)
            return self.codes[index], mask_st, self.nls[index], label
        elif self.task == "cpp":
            label_l = []
            new_nls_l = []
            concept = self.nls[index]
            nls_l = concept.split(",")
            for nls in nls_l:
                child_l = nls.split(self.spliter)
                if len(child_l) < 3:
                    label_l.extend(child_l)
                    new_nls_l.append(nls)
                else:
                    edge_l = self.split_edge_name(child_l[1])
                    edge_str = self.spliter.join(edge_l).lower()

                    new_label = child_l[0] + self.spliter + edge_str + self.spliter + child_l[2]
                    label_l.append(new_label)

            label = ",".join(label_l)
            new_nls = ",".join(new_nls_l)
            return self.codes[index], self.structures[index], new_nls, label
        elif self.task == "nlmp":
            concept = self.nls[index]
            nls_l = concept.split(",")
            new_nls_l = []
            mask_l = []
            for nls in nls_l:
                child_l = nls.split(self.spliter)
                if len(child_l) >= 2:
                    mask_num = len(child_l) // 2
                    random_numbers = random.sample(range(0, len(child_l)), mask_num)
                    for random_number in random_numbers:
                        mask_l.append(child_l[random_number])
                        child_l[random_number] = Vocab.MSK_TOKEN
                    new_child = self.spliter.join(child_l)
                    new_nls_l.append(new_child)
                else:
                    new_nls_l.append(nls)
            new_nls = ",".join(new_nls_l)
            label = " ".join(mask_l)
            return self.codes[index], self.structures[index], new_nls, label
        elif self.task == "mnp":
            code_l = self.codes[index].split(" ")
            func_name = ""
            for index, code in enumerate(code_l):
                if code == "(":
                    func_name = code_l[index-1]
                    code_l[index-1] = Vocab.MSK_TOKEN
                    break
            mask_name_code = " ".join(code_l)

            func_l = self.split_edge_name(func_name)
            func_l = [x.lower() for x in func_l]
            func_l_str = " ".join(func_l)
            old_nls = self.nls[index]
            new_nls = self.nls[index].replace(func_l_str+",", "")


            return mask_name_code, self.structures[index], new_nls, func_name




    def set_task(self, task):
        self.task = task

    def save(self):
        """Save to binary pickle file"""
        path = os.path.join(self.args.dataset_save_dir, f'{self.dataset_name}.pk')

        if os.path.exists(self.args.dataset_save_dir) == False:
            os.makedirs(self.args.dataset_save_dir)

        with open(path, mode='wb') as f:
            pickle.dump(self, f)
        logger.info(f'Dataset saved to {path}')

    def load_dataset_from_dir(self, dataset_dir):
        codes = []
        structures = []
        nls = []
        docs = []

        if self.task == "summarization":
            path = os.path.join(dataset_dir, (self.split+".json"))
            codes, structures, nls, docs = self.parse_json_file(path, lang="java")
        elif self.task == "clone":
            # path = os.path.join(dataset_dir, (self.split + ".txt"))
            codes, structures, nls, docs = self.parse_clone_file(self.split)
        else:
            for file in os.listdir(dataset_dir):
                path = os.path.join(dataset_dir, file)
                if path.endswith(".json"):
                    if file.startswith("py"):
                        lang = "python"
                    else:
                        lang = "java"
                    codes, structures, nls, docs = self.parse_json_file(path, lang=lang)

        return codes, structures, nls, docs

    def parse_kg(self, kg):
        st_l = []
        nl_l = []
        nl_map = {}


        for edges in kg:
            if edges["type"] not in self.st_type and edges["type"] != "related_concept":
                edge_l = self.split_edge_name(edges["type"])
                edge_str = self.spliter.join(edge_l).lower()
                ntc = edges["source"]["label"] + self.spliter + edge_str + self.spliter + edges["target"]["label"]
                # exist_nl.append(edges["source"]["label"])
                # exist_nl.append(edges["target"]["label"])
                # ntc = remove_comments_and_docstrings(ntc, "java")
                ntc = replace_string_literal(ntc)
                ntc = self.remove_punctuation_and_replace_dot(ntc)
                if ntc.lower() not in nl_l:
                    nl_l.append(ntc.lower())

            if edges["type"] in self.st_type:
                if edges["source"]["label"]=="" or edges["target"]["label"]=="":
                    continue
                stc = edges["source"]["label"] + self.spliter + edges["type"] + self.spliter + edges["target"]["label"]
                st_l.append(stc)
            else:
                if edges["type"] == 'related_concept':
                    # if edges["target"]["label"] not in exist_nl:
                    #     nl_l.append(edges["target"]["label"])
                    if edges["source"]["label"] in nl_map.keys():
                        if edges["target"]["label"].lower() not in nl_map[edges["source"]["label"]]:
                            nl_map[edges["source"]["label"]].append(edges["target"]["label"].lower())
                    else:
                        nl_map[edges["source"]["label"]] = [edges["target"]["label"]]

        for nlm in nl_map.keys():
            nlm_token = self.spliter.join(nl_map[nlm])
            # nlm_token = remove_comments_and_docstrings(nlm_token, "java")
            nlm_token = self.remove_punctuation_and_replace_dot(nlm_token)
            nlm_token = replace_string_literal(nlm_token)
            if nlm_token.lower() not in nl_l:
                nl_l.append(nlm_token.lower())


        st_token = self.KG_SEP_TOKEN.join(st_l)
        nl_token = ",".join(nl_l)
        return st_token, nl_token

    def remove_punctuation_and_replace_dot(self, text):
        # 使用正则表达式去掉所有标点符号
        text = re.sub(r'[^\w\s.]', '', text)
        # 将 "." 替换为空格
        text = text.replace('.', ' ')
        return text

    def parse_json_file(self, file, lang):
        """
        Parse a dataset file where each line is a json string representing a sample.

        Args:
            file (str): The file path
            lang (str): Source code language

        Returns:
            (list[str], list[str], list[str], list[str], List[str]):
                - List of source codes
                - List of tokenized codes
                - List of split method names
                - List of tokenized codes with method name replaced with ``f``
                - List of docstring strings, not every sample has it

        """

        codes = []
        structures = []
        nls = []
        docs = []

        with open(file, encoding='ISO-8859-1') as f:
            lines = f.readlines()
            print("loading dataset:")
            for line in tqdm(lines):
                # print(line)
                data = json.loads(line.strip())
                doc = data["doc"]
                st, nl = self.parse_kg(data["kg"])

                source = data['code'].strip()
                source = remove_comments_and_docstrings(source, lang)
                print(source)
                source = replace_string_literal(source)
                code = tokenize_source(source=source, lang=lang)
                codes.append(code)

                code_l = code.split(" ")
                func_name = ""
                for index, code in enumerate(code_l):
                    if code == "(":
                        func_name = code_l[index - 1]
                        break
                func_name_l = self.split_edge_name(func_name)
                func_name_nl = " ".join(func_name_l)
                if func_name_nl.lower() not in nl:
                    nl += ","
                    nl += func_name_nl


                structures.append(st)
                nls.append(nl)
                docs.append(doc)

        return codes, structures, nls, docs

    def parse_clone_file(self):
        json_file = os.path.join(self.dataset_dir, "data.json")
        file = os.path.join(self.dataset_dir, (self.split + ".txt"))

        codes_1 = []
        codes_2 = []
        labels = []

        json_data = {}

        with open(json_file, encoding='ISO-8859-1') as jf:
            lines = jf.readlines()
            print("loading dataset:")
            for line in tqdm(lines):
                # print(line)
                data = json.loads(line.strip())
                json_data[data["idx"]] = data["func"]

        with open(file, encoding='ISO-8859-1') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                ll = line.split(" ")
                codes_1.append(ll[0])
                codes_2.append(ll[1])
                labels.append(ll[2])


def init_dataset(args, task=None, split=None, load_if_saved=True, mode="pretrain", language="java"):
    name = '.'.join([sub_name for sub_name in [mode, task, language, split] if sub_name is not None])
    if load_if_saved:
        path = os.path.join(args.dataset_save_dir, f'{name}.pk')
        print(path)
        if os.path.exists(path) and os.path.isfile(path):
            logger.info(f'Trying to load saved binary pickle file from: {path}')
            with open(path, mode='rb') as f:
                obj = pickle.load(f)
            assert isinstance(obj, KGCodeDataset)
            obj.args = args
            logger.info(f'Dataset instance loaded from: {path}')
            # print_paths(obj.paths)
            return obj
    dataset = KGCodeDataset(args=args, task=task)
    dataset.save()
    return dataset

def print_paths(paths):
    """
    Print paths.

    Args:
        paths (dict): Dict mapping path group to path string or list of path strings.

    """
    logger.info('Dataset loaded from these files:')
    for key, value in paths.items():
        if isinstance(value, list):
            for v in value:
                logger.info(f'  {key}: {v}')
        else:
            logger.info(f'  {key}: {value}')