import os
import json
import pickle
import random
import logging
import re

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from common.data_utils import tokenize_source, remove_comments_and_docstrings, replace_string_literal
from common.vocab import Vocab

# from pretrain.vocab.vocab import Vocab

logger = logging.getLogger(__name__)


class KGCodeDataset(Dataset):
    spliter = " "
    KG_SEP_TOKEN = " " + Vocab.KG_SEP_TOKEN + " "

    def __init__(self, args, task, split=None):
        self.args = args
        self.task = task
        self.dataset_dir = os.path.join(args.dataset_root)
        self.split = split
        self.codes = []
        self.structures = []
        self.nls = []
        self.docs = []
        self.st_type = [
            'type_of',
            "control_dependency",
            "data_dependency",
            "has_method",
            "has_property",
            "assignment"
        ]
        self.dataset_name = "KGCode"
        if self.task == "clone" or self.task == "clone2":
            self.codes1, self.sts1, self.docs1, self.codes2, self.sts2, self.docs2, self.labels = self.load_clone_dataset()
        else:
            self.codes, self.structures, self.nls, self.docs = self.load_dataset_from_dir(dataset_dir=self.dataset_dir)

    def split_edge_name(self, name):
            # 处理 CamelCase
        return re.sub('([a-z])([A-Z])', r'\1 \2', name).split()


    def __len__(self):
        if self.task == "clone" or self.task == "clone2":
            return len(self.codes1)
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
        indices = random.sample(range(len(self.labels)), int(len(self.labels) * ratio))
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
        elif self.task == "summarization":
            return self.codes[index], self.structures[index], self.nls[index], self.docs[index]
        elif self.task == "clone":
            return self.codes1[index], self.sts1[index], "", self.codes2[index], self.sts2[index], "", self.labels[index]
        elif self.task == "clone2":
            return self.codes1[index], self.sts1[index], self.docs1[index], self.codes2[index], self.sts2[index], \
            self.docs2[index], self.labels[index]

    def set_task(self, task):
        self.task = task

    def save(self):
        """Save to binary pickle file"""
        path = os.path.join(self.args.dataset_save_dir, f'{self.dataset_name+"."+self.split}.pk')

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
            codes, structures, nls, docs = self.parse_json_file(path)
        else:
            for file in os.listdir(dataset_dir):
                path = os.path.join(dataset_dir, file)
                if path.endswith(".json"):
                    codes, structures, nls, docs = self.parse_json_file(path)

        return codes, structures, nls, docs

    def remove_punctuation_and_replace_dot(self, text):
        # 使用正则表达式去掉所有标点符号
        text = re.sub(r'[^\w\s.]', '', text)
        # 将 "." 替换为空格
        text = text.replace('.', ' ')
        return text

    def load_clone_dataset(self):
        codes1, sts1, docs1, codes2, sts2, docs2, labels = self.parse_clone_file()
        return codes1, sts1, docs1, codes2, sts2, docs2, labels

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
                if edges["source"]["label"] == "" or edges["target"]["label"] == "":
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

    def parse_json_file(self, file):
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
                code = data["code"]
                doc = data["doc"]
                st, nl = self.parse_kg(data["kg"])

                source = data['code'].strip()
                source = source.replace("\t", " ")
                source = remove_comments_and_docstrings(source, "java")
                source = replace_string_literal(source)
                code = tokenize_source(source=source, lang="java")
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
        sts_1 = []
        sts_2 = []
        docs_1 = []
        docs_2 = []
        labels = []

        json_data = {}

        with open(json_file, encoding='ISO-8859-1') as jf:
            lines = jf.readlines()
            print("loading dataset:")
            for line in tqdm(lines):
                # print(line)
                data = json.loads(line.strip())
                st, nl = self.parse_kg(data["kg"])

                source = data['code'].strip()
                source = remove_comments_and_docstrings(source, "java")
                source = replace_string_literal(source)
                code = tokenize_source(source=source, lang="java")
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

                json_data[data["idx"]] = {
                    "code" : " ".join(code_l),
                    "st" : st,
                    "nl" : nl,
                }

        with open(file, encoding='ISO-8859-1') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                try:
                    ll = line.split("\t")
                    if ll[0] not in json_data.keys() or ll[1] not in json_data.keys():
                        continue

                    codes_1.append(json_data[ll[0]]["code"])
                    codes_2.append(json_data[ll[0]]["code"])

                    st1 = json_data[ll[0]]["st"]
                    sts_1.append(st1)
                    st2 = json_data[ll[1]]["st"]
                    sts_2.append(st2)

                    doc1 = json_data[ll[0]]["nl"]
                    docs_1.append(doc1)
                    doc2 = json_data[ll[1]]["nl"]
                    docs_2.append(doc2)

                    label = ll[2].replace("\n", "")
                    labels.append(int(label))
                except Exception as e:
                    # print(e)
                    continue
                # codes_1.append(ll[0])
                # codes_2.append(ll[1])
                # labels.append(ll[2])
        return codes_1, sts_1, docs_1, codes_2, sts_2, docs_2, labels


def init_dataset(args, task=None, split=None, load_if_saved=True, mode="summarization", language="java"):
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
    dataset = KGCodeDataset(args=args, task=task, split=split)
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