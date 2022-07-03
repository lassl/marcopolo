import pickle
from typing import Callable, Optional

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from marcopolo.dpr.datasets.marco import MARCO_Dataset

class DataPipeline(LightningDataModule):
    def __init__(self, hf_datasets, tokenizer, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.dataset_builder = MARCO_Dataset
        self.hf_datasets = hf_datasets
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.hf_datasets,
                self.tokenizer,
                "TRAIN",
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.hf_datasets,
                self.tokenizer,
                "VALID"
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                self.hf_datasets,
                self.tokenizer,
                "TEST",
            )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=True,
            collate_fn = self.test_dataset._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn = self.test_dataset._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            collate_fn = self.test_dataset._collate_fn,
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, hf_datasets, tokenizer, split) -> Dataset:
        dataset = dataset_builder(hf_datasets, tokenizer, split)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size, num_workers, drop_last, shuffle, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last = drop_last, 
            shuffle = shuffle,
            **kwargs
        )