import torch 
import numpy as np
import pandas as pd
import json
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from typing import Any, Tuple


from transformers import AutoTokenizer



random.seed(42)

class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=512, output_embed_dim=128, num_layers=3, num_heads=8):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embed_dim)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(embed_dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers,
            norm=torch.nn.LayerNorm([embed_dim]),
            enable_nested_tensor=False
        )
        self.projection = torch.nn.Linear(embed_dim, output_embed_dim)

    def forward(self, tokenizer_ouput):
        if tokenizer_ouput['input_ids'].dim() == 3:
            tokenizer_ouput['input_ids'] = tokenizer_ouput['input_ids'].squeeze(1)
        if tokenizer_ouput['attention_mask'].dim() == 3:
            tokenizer_ouput['attention_mask'] = tokenizer_ouput['attention_mask'].squeeze(1)
        x = self.embedding_layer(tokenizer_ouput['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_ouput['attention_mask'].logical_not())
        cls_embed = x[:, 0, :] 
        return self.projection(cls_embed)
    


with open('./dataset/train-v2.0.json') as f:
    train_dataset = json.load(f)

with open('./dataset/dev-v2.0.json') as f:
    dev_dataset = json.load(f)


# prepare dataset squad

def prepare_dataset(dataset):
    dataset = []
    for data in dataset['data']:
        title = data['title']
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                if len(qa['answers']) == 0:
                    answer = qa['plausible_answers'][0]['text']
                else:
                    answer = qa['answers'][0]['text']
                dataset += [
                    {
                        'question': question,
                        'answer': answer,
                        'title': title
                    }
                ]
    return dataset


dataset = prepare_dataset(train_dataset)

random.shuffle(dataset)

train_size = int(0.8*len(dataset))
train_dataset = dataset[:train_size]
dev_size = len(dataset) - train_size    
test_size = int(0.5*dev_size)
dev_dataset = dataset[train_size:train_size+test_size]
test_dataset = dataset[train_size+test_size:]




class QADataset(Dataset):
    def __init__(self, dataset, max_seq_len=512) -> None:
        super().__init__()
        self.dataset = dataset 
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index) -> Tuple[str, str]:
        question = self.dataset[index]['question']
        answer = self.dataset[index]['answer']
        question_tok = self.tokenizer(question, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_seq_len)
        answer_tok = self.tokenizer(answer, padding='max_length', truncation=True, return_tensors='pt', max_length=self.max_seq_len)
        return {
            'question': question,
            'answer': answer,
            'question_tok': question_tok, 
            'answer_tok': answer_tok 
        }

if __name__ == '__main__':

    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 8
    NUM_EPOCHS = 10
    train = QADataset(train_dataset)
    dev = QADataset(dev_dataset)
    test = QADataset(test_dataset)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    n_iters = len(train_dataset) // batch_size + 1
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size, num_layers=3, num_heads=8).to(device)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size, num_layers=3, num_heads=8).to(device)

    # 
    optimizer = optim.AdamW(list(question_encoder.parameters()) + list(answer_encoder.parameters()), lr=1e-5)
    fn_loss = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        running_loss = []
        for idx, data_batch in enumerate(train_loader):
            question_tok = data_batch['question_tok'].to(device)
            answer_tok = data_batch['answer_tok'].to(device)
            if idx == 0 and epoch == 0:
                print(question_tok['input_ids'].shape, answer_tok['input_ids'].shape)
                print(question_tok['attention_mask'].shape, answer_tok['attention_mask'].shape) 
            question_embed = question_encoder(question_tok)
            answer_embed = answer_encoder(answer_tok)
            similarity_scores = question_embed @ answer_embed.T
            target = torch.arange(question_embed.shape[0], dtype=torch.long).to(device)
            loss = fn_loss(similarity_scores, target)
            if idx == 0 and epoch == 0:
                print(f"Loss at init = {loss.item()}")
            running_loss.append(loss.log10().item())
            if idx == n_iters-1:
                print(f"Epoch {epoch}, loss = ", np.mean(running_loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # evaluate on the dev set
        with torch.no_grad():
            dev_running_loss = []
            for idx, data_batch in enumerate(dev_loader):
                question_tok = data_batch['question_tok'].to(device)
                answer_tok = data_batch['answer_tok'].to(device)
                question_embed = question_encoder(question_tok)
                answer_embed = answer_encoder(answer_tok)
                similarity_scores = question_embed @ answer_embed.T
                target = torch.arange(question_embed.shape[0], dtype=torch.long).to(device)
                loss = fn_loss(similarity_scores, target)
                dev_running_loss.append(loss.log10().item())
            print(f"Dev loss = ", np.mean(running_loss))
