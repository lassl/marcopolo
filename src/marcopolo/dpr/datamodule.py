from typing import Any, Dict, List

from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PreTrainedTokenizer, AutoTokenizer
from abc import ABC, abstractmethod


class DPRDataModule:
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        accelerator: Accelerator,
        max_seq_length: int,
        # per_device_train_batch_size: int,
        # per_device_eval_batch_size: int,
        test_size: float,
        num_process: int,
        num_workers: int,
        # seed: int,
        args
    ):
        self.tokenizer = tokenizer
        self.data_collator = CustomDataCollator(tokenizer)
        self.accelerator = accelerator
        self.max_seq_length = min(max_seq_length, tokenizer.model_max_length)
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.per_device_eval_batch_size = args.per_device_eval_batch_size
        self.test_size = test_size
        self.num_process = num_process
        self.num_workers = num_workers
        self.seed = args.seed
        dataset = dataset.train_test_split(test_size=test_size, seed=args.seed)
        self.train_dataset, self.valid_dataset = dataset["train"], dataset["test"]

        # with self.accelerator.main_process_first():
        remove_columns = self.train_dataset.column_names
        self.processed_train_dataset = self.train_dataset.map(
            lambda example: self._tokenize(example),
            remove_columns=remove_columns,
            keep_in_memory=True,
            batched=True,
            num_proc=self.num_process,
            desc="processing train dataset",
        )
        self.processed_valid_dataset = self.valid_dataset.map(
            lambda example: self._tokenize(example),
            remove_columns=remove_columns,
            keep_in_memory=True,
            batched=True,
            num_proc=self.num_process,
            desc="processing valid dataset",
        )
        # self.accelerator.wait_for_everyone()

    def _tokenize(self, example):
        # q_result = self.tokenizer(example["query"], truncation=True)
        # positive_passages = [
        #     passage[0]["text"] for passage in example["positive_passages"]
        # ]
        # p_result = self.tokenizer(positive_passages, truncation=True)
        # negative_passages = [
        #     passage[0]["text"] for passage in example["negative_passages"]
        # ]
        # n_result = self.tokenizer(negative_passages, truncation=True)

        q_result = self.tokenizer.prepare_for_model(example["query"])
        p_result = self.tokenizer.prepare_for_model(example["positives"][0])
        n_result = self.tokenizer.prepare_for_model(example["negatives"][0])
        
        result = dict()
        for key, value in q_result.items():
            result["q_" + key] = value
        for key, value in p_result.items():
            result["p_" + key] = value
        for key, value in n_result.items():
            result["n_" + key] = value
        return result

    def train_dataloader(self):
        return DataLoader(
            self.processed_train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )

    def valid_dataloader(self):
        return DataLoader(
            self.processed_valid_dataset,
            batch_size=self.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
        )



class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        q_features = [
            {key[2:]: value for key, value in feature.items() if key.startswith("q_")} for feature in features
        ]
        p_features = [
            {key[2:]: value for key, value in feature.items() if key.startswith("p_")} for feature in features
        ]
        n_features = [
            {key[2:]: value for key, value in feature.items() if key.startswith("n_")} for feature in features
        ]
        q_batch = self.tokenizer.pad(
            q_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        p_batch = self.tokenizer.pad(
            p_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        n_batch = self.tokenizer.pad(
            n_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = dict()
        for key, value in q_batch.items():
            batch["q_" + key] = value
        for key, value in p_batch.items():
            batch["p_" + key] = value
        for key, value in n_batch.items():
            batch["n_" + key] = value
        return batch

