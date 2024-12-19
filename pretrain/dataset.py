import os
import json
import pickle
import random
import logging
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from vocab import Vocab

# from pretrain.vocab.vocab import Vocab

logger = logging.getLogger(__name__)


class KGCodeDataset(Dataset):
    spliter = ";"

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
        self.codes, self.structures, self.nls, self.docs = self.load_dataset_from_dir(dataset_dir=self.dataset_dir)


    def __len__(self):
        return len(self.codes)

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
             st_l = structure.split(Vocab.KG_SEP_TOKEN)
             for st in st_l:
                child_l = st.split(self.spliter)
                if len(child_l) < 3:
                    continue
                label_l.append(child_l[1])
                new_st = child_l[0] + self.spliter + Vocab.MSK_TOKEN + self.spliter + child_l[2]
                new_st_l.append(new_st)
             mask_st = Vocab.KG_SEP_TOKEN.join(new_st_l)
             label = ",".join(label_l)
             return self.codes[index], mask_st, self.nls[index], label
        elif self.task == "nlp":
            return self.codes[index], self.structures[index], self.nls[index], self.docs[index]
        elif self.task == "cgp":
            is_graph = random.random() < 0.5
            if is_graph:
                return self.codes[index], self.structures[index], self.nls[index], 1
            else:
                other_graph = self.structures[random.randint(0, len(self.structures) - 1)]
                while other_graph == self.structures[index]:
                    other_graph = self.structures[random.randint(0, len(self.structures) - 1)]
                return self.codes[index], other_graph, self.nls[index], 0

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
            codes, structures, nls, docs = self.parse_json_file(path)
        else:
            for file in os.listdir(dataset_dir):
                path = os.path.join(dataset_dir, file)
                if path.endswith(".json"):
                    codes, structures, nls, docs = self.parse_json_file(path)

        return codes, structures, nls, docs

    def parse_kg(self, kg):
        st_l = []
        nl_l = []
        nl_map = {}


        for edges in kg:
            if edges["type"] not in self.st_type and edges["type"] != "related_concept":
                ntc = edges["source"]["label"] + self.spliter + edges["type"] + self.spliter + edges["target"]["label"]
                # exist_nl.append(edges["source"]["label"])
                # exist_nl.append(edges["target"]["label"])
                nl_l.append(ntc)

            if edges["type"] in self.st_type:
                stc = edges["source"]["label"] + self.spliter + edges["type"] + self.spliter + edges["target"]["label"]
                st_l.append(stc)
            else:
                if edges["type"] == 'related_concept':
                    # if edges["target"]["label"] not in exist_nl:
                    #     nl_l.append(edges["target"]["label"])
                    if edges["source"]["label"] in nl_map.keys():
                        nl_map[edges["source"]["label"]].append(edges["target"]["label"])
                    else:
                        nl_map[edges["source"]["label"]] = [edges["target"]["label"]]

        for nlm in nl_map.keys():
            nlm_token = self.spliter.join(nl_map[nlm])
            nl_l.append(nlm_token)


        st_token = Vocab.KG_SEP_TOKEN.join(st_l)
        nl_token = Vocab.KG_SEP_TOKEN.join(nl_l)
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

                codes.append(code)
                structures.append(st)
                nls.append(nl)
                docs.append(doc)

        return codes, structures, nls, docs


def init_dataset(args, task=None, split=None, load_if_saved=True):
    name = "kgcode.pretrain.codesearchnet.java"
    if load_if_saved:
        path = os.path.join(args.dataset_save_dir, f'{name}.pk')
        if os.path.exists(path) and os.path.isfile(path):
            logger.info(f'Trying to load saved binary pickle file from: {path}')
            with open(path, mode='rb') as f:
                obj = pickle.load(f)
            assert isinstance(obj, KGCodeDataset)
            obj.args = args
            logger.info(f'Dataset instance loaded from: {path}')
            print_paths(obj.paths)
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