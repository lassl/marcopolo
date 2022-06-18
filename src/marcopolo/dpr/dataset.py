import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class Preprocess:
    def __init__(self, tokenizer, args):
        self.args = args
        self.tokenizer = tokenizer

    def make_dataset(self, data) -> TensorDataset:

        tokenized_query = self.tokenizer(
            list(data['query'])[:self.args.samples], padding=True, truncation=True, max_length=self.args.max_length, return_tensors="pt"
        )

        tokenized_passage = self.tokenizer(
            list(data['pos_passage'])[:self.args.samples], padding=True, truncation=True, max_length=self.args.max_length, return_tensors="pt"
        )

        print(f'tokenized query length: {len(tokenized_query["input_ids"])}')
        print(f"tokenized passage length: {len(tokenized_passage['input_ids'])}")
        print(tokenized_passage["input_ids"].shape)

        dataset = TensorDataset(
            tokenized_query["input_ids"],
            tokenized_query["attention_mask"],
            tokenized_query["token_type_ids"],
            tokenized_passage["input_ids"],
            tokenized_passage["attention_mask"],
            tokenized_passage["token_type_ids"],
        )

        return dataset

    def make_iteration(self, dataset):

        return DataLoader(dataset, shuffle= True, batch_size= self.args.batch_size)

