"""
  TODO:
  1. data load해서 in-batch sample로 구성 -> pos sample, neg sample, query 3개로 Dataset을 구성

"""
import argparse
from xmlrpc.client import boolean

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig
from dataset import DataModule
from model import DPR
from trainer import Trainer

def get_config():
  parser = argparse.ArgumentParser()

  """utils"""
  parser.add_argument("--wandb", type=bool, default=True, help="wandb on/off")

  """model option"""
  parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
  parser.add_argument("--query_model_name", type=str, default="roberta-base")
  parser.add_argument("--passage_model_name", type=str, default="roberta-base")

  """data"""
  parser.add_argument("--samples", type=int, default=10000)  # choose data sample size
  parser.add_argument("--negative", type=int, default=3)  # neg_sample count
  parser.add_argument("--per_device_train_batch_size", type=int, default=8)
  parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
  parser.add_argument("--max_length", type=int, default=256)
  parser.add_argument("--test_size", type=float, default=0.2)
  parser.add_argument("--num_workers", type=int, default=2)

  """train"""
  parser.add_argument("--num_train_epochs", type=int, default=20)
  parser.add_argument("--learning_rate", type=float, default=1e-5)
  parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
  parser.add_argument("--logging_steps", type=int, default=50)
  parser.add_argument("--num_warmup_steps", type=int, default=0)

  parser.add_argument("--epsilon", type=float, default=1e-8)


  args = parser.parse_args()

  return args




if __name__ == "__main__":

  args= get_config()

  print(f'===load dataset===')  
  data= load_dataset('Tevatron/msmarco-passage')

  print(f'===load model && tokenizer===')
  query_model= DPR(args.query_model_name)
  passage_model= DPR(args.passage_model_name)
  tokenizer= AutoTokenizer.from_pretrained(args.passage_model_name)

  print(f'===load data module===')
  datamodule= DataModule(data, tokenizer, args)

  print(f'===initialize trainer===')
  trainer= Trainer(query_model, passage_model, datamodule, args)

  print(f'===training start===')
  trainer.train()
  print(f'===training end===')


