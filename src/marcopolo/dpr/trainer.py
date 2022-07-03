

from tqdm import tqdm

import os
import torch
from transformers import get_linear_schedule_with_warmup

import wandb

class Trainer:
    def __init__(
        self,
        query_model,
        passage_model,
        datamodule,
        args,
    ):
        self.args= args

        self.query_model= query_model
        self.passage_model= passage_model
        self.criterion= torch.nn.CrossEntropyLoss()

        self.train_loader= datamodule.trainloader
        self.valid_loader= datamodule.validloader
        
        self.optimizer= self.get_optimizer()
        self.scheduler= get_linear_schedule_with_warmup(
        optimizer=self.optimizer, num_warmup_steps= self.args.num_warmup_steps, num_training_steps= len(self.train_loader)* self.args.num_train_epochs
        )

        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.query_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-4,
            },
            {
                'params': [p for n, p in self.passage_model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return optimizer
    
    def save_model(self):
        if not os.path.exists('./model'):
            os.makedirs('./model')

        if not os.path.exists("./model/query_model") and not os.path.exists("./model/passage_model"):
            os.makedirs("./model/query_model")
            os.makedirs("./model/passage_model")

        torch.save(self.query_model.state_dict(), f"./model/query_model/query_model_{self.epoch}.pt")
        torch.save(self.passage_model.state_dict(), f"./model/passage_model/passage_model_{self.epoch}.pt")

    def train(self):
        if self.args.wandb:
            run = wandb.init(project="ms-marco", entity="rockmiin", name="dpr")

        best_acc= 0
        for epoch in range(self.args.num_train_epochs):
            self.epoch= epoch
            self.train_one_epoch()
            valid_loss, valid_acc= self.validation()

            print(f'epoch: {self.epoch} valid loss: {valid_loss}, valid acc: {valid_acc}')

            if self.args.wandb:
                wandb.log({"valid/acc": valid_acc, "valid/loss": valid_loss})

            if best_acc < valid_acc:
                self.save_model()
                best_acc= valid_acc
        
        if self.args.wandb:
            run.finish()
    
    def train_one_epoch(self):
        
        self.query_model.train()
        self.passage_model.train()

        train_loss, train_acc= 0, 0
        acc_steps, loss_steps= 0, 0

        pbar= tqdm(enumerate(self.train_loader), total= len(self.train_loader))
        for step, batch in pbar:
            pbar.set_description(f'epoch: {self.epoch}')

            if torch.cuda.is_available():
                batch= tuple(t.to(self.device) for t in batch)
                self.query_model.to(self.device)
                self.passage_model.to(self.device)

            q_inputs= {"input_ids": batch[0], "attention_mask": batch[1]}
            pos_inputs= {"input_ids": batch[2], "attention_mask": batch[3]}
            neg_inputs= {
                "input_ids": batch[4].view(batch[0].shape[0]* self.args.negative, -1), 
                "attention_mask": batch[5].view(batch[0].shape[0]* self.args.negative, -1)
            }

            q_outputs= self.query_model(**q_inputs)
            pos_outputs= self.passage_model(**pos_inputs)
            neg_outputs= self.passage_model(**neg_inputs)

            ctx_outputs= torch.cat([pos_outputs, neg_outputs], dim= 0) # (1+ neg) * emb

            sim_scores= torch.matmul(q_outputs, ctx_outputs.T)
            targets= torch.arange(0, batch[0].shape[0]).to(self.device)





            # pos_outputs= pos_outputs.view(batch[0].shape[0], 1, -1)
            # neg_outputs= neg_outputs.view(batch[0].shape[0], self.args.negative, -1)
            # print(f'query shape : {q_outputs.shape}')
            # print(pos_outputs.shape, neg_outputs.shape)
            
            # q_outputs= q_outputs.view(batch[0].shape[0], 1, -1) # batch * 1 * emb
            # ctx_outputs= torch.cat([pos_outputs, neg_outputs], dim= 0) # batch * (pos+ neg) * emb

            # print('ctx shape', ctx_outputs.shape)
            # sim_scores= torch.bmm(q_outputs, torch.transpose(ctx_outputs, 1, 2)).squeeze() 
            # print('sim scores', sim_scores.shape)
            # targets= torch.arange(0, batch[0].shape[0]).to(self.device)




            # ctx_outputs= torch.cat([pos_outputs, neg_outputs], dim= 1) # batch * (pos+ neg) * emb
            # ctx_outputs= ctx_outputs.view(batch[0].shape[0] * (self.args.negative+1), -1)

            # sim_scores= torch.matmul(q_outputs, ctx_outputs.T).squeeze()
            # # print(sim_scores.shape)
            # sim_scores= sim_scores.view(batch[0].shape[0], -1)
            # targets= torch.arange(0, batch[0].shape[0]).to(self.device)




            preds= torch.argmax(sim_scores, axis= 1)
            loss= self.criterion(sim_scores, targets)

            loss= loss/ self.args.gradient_accumulation_steps
            loss.backward()
            
            train_loss+= loss.item()
            train_acc+= torch.sum(preds== targets)
            acc_steps+= batch[0].shape[0]
            loss_steps+= 1

            if (step+ 1)% self.args.gradient_accumulation_steps== 0 or (step+ 1)== len(self.train_loader): # gradient accumulation
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            if (step+ 1)% self.args.logging_steps== 0 or (step+ 1)== len(self.train_loader):
                if acc_steps == 0 : acc_steps= self.args.epsilon
                
                train_acc= train_acc / acc_steps
                train_loss= train_loss / loss_steps

                print(f'epoch: {self.epoch}, train loss: {train_loss}, train acc: {train_acc}')

                if self.args.wandb:
                    wandb.log({"train/acc": train_acc, "train/loss": train_loss})
                
                train_loss, train_acc= 0, 0
                acc_steps, loss_steps= 0, 0
    
    def validation(self):

        self.query_model.eval()
        self.passage_model.eval()
        
        valid_loss, valid_acc= 0, 0

        with torch.no_grad():
            pbar= tqdm(enumerate(self.valid_loader), total= len(self.valid_loader))
            for step, batch in pbar:

                if torch.cuda.is_available():
                    batch= tuple(t.to(self.device) for t in batch)
                    self.query_model.to(self.device)
                    self.passage_model.to(self.device)
            
                q_inputs= {"input_ids": batch[0], "attention_mask": batch[1]}
                pos_inputs= {"input_ids": batch[2], "attention_mask": batch[3]}
                neg_inputs= {
                    "input_ids": batch[4].view(batch[0].shape[0]* self.args.negative, -1), 
                    "attention_mask": batch[5].view(batch[0].shape[0]* self.args.negative, -1)
                }

                q_outputs= self.query_model(**q_inputs)
                pos_outputs= self.passage_model(**pos_inputs)
                neg_outputs= self.passage_model(**neg_inputs)

                ctx_outputs= torch.cat([pos_outputs, neg_outputs], dim= 0) # (1+ neg) * emb

                sim_scores= torch.matmul(q_outputs, ctx_outputs.T)
                targets= torch.arange(0, batch[0].shape[0]).to(self.device)

                loss= self.criterion(sim_scores, targets)
                preds= torch.argmax(sim_scores, axis= 1)

                valid_loss+= loss.item()
                valid_acc+= torch.sum(preds== targets)
        
        valid_loss_= valid_loss / len(self.valid_loader)
        valid_acc_= valid_acc / len(self.valid_loader.dataset)

        return valid_loss_, valid_acc_








        



