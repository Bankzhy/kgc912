import json
import random
import logging

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

def convert_clone_examples_to_features(idx, item):
    # example, example_index, tokenizer, args = item
    url1, url2, label, tokenizer, args, cache, url_to_code = item
    source = url_to_code[url1]
    target = url_to_code[url2]

    source_str = "{}: {}".format(args.task, source)
    target_str = "{}: {}".format(args.task, target)

    # if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
    #     source_str = "{}: {}".format(args.task, source)
    #     target_str = "{}: {}".format(args.task, target)
    # else:
    #     source_str = source
    #     target_str = target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(idx, source_ids, label, url1, url2)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def ty__init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class T5TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train'):
        self.examples = []
        self.args = args
        index_filename = file_path

        # load index
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code = {}
        with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                url_to_code[js['idx']] = js['func']

        # load code function according to index
        data = []
        cache = {}
        f = open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line = line.strip()
                url1, url2, label = line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label == '0':
                    label = 0
                else:
                    label = 1
                data.append((url1, url2, label, tokenizer, args, cache, url_to_code))

        # only use 10% valid data to keep best model
        if 'valid' in file_path:
            data = random.sample(data, int(len(data) * 0.1))

        # convert example to input features
        self.examples = [convert_clone_examples_to_features(idx, x) for idx, x in tqdm(enumerate(data), total=len(data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        return (torch.tensor(self.examples[item].source_ids), torch.tensor(self.examples[item].label))

