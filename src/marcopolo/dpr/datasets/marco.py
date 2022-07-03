import os
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from typing import Callable, List, Dict, Any
from torch.utils.data import Dataset

class MARCO_Dataset(Dataset):
    def __init__(self, hf_datasets, tokenizer, split):
        self.split = split
        self.tokenizer = tokenizer
        msmarco = hf_datasets.train_test_split(test_size=int(len(hf_datasets)*0.1), seed=42)
        self.train_dataset, self.valid_dataset = msmarco["train"], msmarco["test"]
        self.get_filelist()
    
    def get_filelist(self):
        if self.split == "TRAIN":
            self.fl = self.train_dataset
        elif self.split == "VALID":
            self.fl = self.valid_dataset
        elif self.split == "TEST":
            self.fl = self.valid_dataset
        else:
            raise ValueError(f"Unexpected split name: {self.split}")

    def _collate_fn(self, example):
        query = example["query"]
        positive_passages = [passage["text"] for passage in example["positive_passages"] if passage["text"]]
        pos = random.choice(positive_passages)
        neg = [passage["text"] for passage in example["negative_passages"] if passage["text"]]
        query = self.tokenizer(query, truncation=True)
        pos = self.tokenizer(pos, truncation=True)
        neg = self.tokenizer(neg, truncation=True)
        return query, pos, neg

    def __getitem__(self, index):
        return self.fl[index]
    
    def __len__(self):
        return len(self.fl)