import argparse
from typing import Optional

import torch
from torch.utils.data.dataset import Dataset
from transformers import Seq2SeqTrainer, Trainer
from torch.utils.data.dataloader import DataLoader
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")


def collate_fn(batch, args):
    prefix = "Summarize Ruby: "
    model_inputs = {}
    code_raw, ast_raw, nl_raw, name_raw = map(list, zip(*batch))
    inputs = [prefix + code for code in code_raw]
    model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding="max_length", truncation=True)
    # encode the summaries
    labels = tokenizer(name_raw, max_length=args.max_target_length, padding="max_length", truncation=True).input_ids
    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"], dtype=torch.long)
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"], dtype=torch.long)
    model_inputs["labels"] = torch.tensor(model_inputs["labels"], dtype=torch.long)
    return model_inputs


class T5Trainer(Seq2SeqTrainer):

    def __init__(self, main_args: argparse.Namespace, **kwargs):
        super(T5Trainer, self).__init__(**kwargs)
        self.main_args = main_args

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.main_args.batch_size,
                          shuffle=True,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset:
            self.eval_dataset = eval_dataset
        return DataLoader(dataset=self.eval_dataset,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args))

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return DataLoader(dataset=test_dataset,
                          batch_size=self.main_args.eval_batch_size,
                          collate_fn=lambda batch: collate_fn(batch,
                                                              args=self.main_args))

    def set_task(self, task):
        self.task = task
