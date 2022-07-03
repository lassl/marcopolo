from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

import torch
import numpy as np


class DataModule:
  def __init__(
    self,
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    args
  ):

    self.args= args
    self.tokenizer= tokenizer

    dataset= dataset['train'].train_test_split(test_size= self.args.test_size, seed= args.seed)
    self.trainset, self.validset= dataset['train'][:self.args.samples], dataset['test'][:self.args.samples]
    self.tokenized_trainset= self._tokenize(self.trainset)
    self.tokenized_validset= self._tokenize(self.validset)

    self.trainloader= self.train_loader()
    self.validloader= self.valid_loader()


  def _tokenize(self, sample): 
    positive_passages= [
      passage[0]['text'] if isinstance(passage, list) else passage['text']
      for passage in sample['positive_passages']
    ]

    negative_passages= []
    for passage in sample['negative_passages']:
      if len(passage) >= self.args.negative:
        batch_negative_sample= [passage[i]['text'] for i in range(self.args.negative)]
      else:
        diff= self.args.negative- len(passage)
        batch_negative_sample= [passage[i]['text'] for i in range(len(passage))]
        batch_negative_sample.extend([passage[0]['text']] *diff) # add sample
        # print(f'batch neg sample:{len(batch_negative_sample)}')

      negative_passages.extend(batch_negative_sample)
    
    tokenized_query= self.tokenizer(
      sample['query'],
      padding= 'max_length',
      truncation= True,
      max_length= self.args.max_length,
      return_tensors= 'pt'
    )
    
    pos_tokenized_passages= self.tokenizer(
      positive_passages,
      padding='max_length', 
      truncation=True, 
      max_length=self.args.max_length, 
      return_tensors="pt"
    )
    
    neg_tokenized_passages= self.tokenizer(
      negative_passages, 
      padding='max_length', 
      truncation=True, 
      max_length=self.args.max_length, 
      return_tensors="pt"
    )

    neg_tokenized_passages['input_ids']= neg_tokenized_passages['input_ids'].view(-1, self.args.negative, self.args.max_length)
    neg_tokenized_passages['attention_mask']= neg_tokenized_passages['input_ids'].view(-1, self.args.negative, self.args.max_length)

    # print(tokenized_query['input_ids'].shape)
    # print(pos_tokenized_passages['input_ids'].shape)
    # print(neg_tokenized_passages['input_ids'].shape)

    return TensorDataset(
      tokenized_query['input_ids'],
      tokenized_query['attention_mask'],
      pos_tokenized_passages['input_ids'],
      pos_tokenized_passages['attention_mask'],
      neg_tokenized_passages['input_ids'],
      neg_tokenized_passages['attention_mask']
    )
    
  def train_loader(self):
    return DataLoader(
      self.tokenized_trainset,
      batch_size= self.args.per_device_train_batch_size,
      shuffle= True,
    )
  
  def valid_loader(self):
    return DataLoader(
      self.tokenized_validset,
      batch_size= self.args.per_device_eval_batch_size,
      shuffle= False,
    )   






