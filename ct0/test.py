import json
import torch
from transformers import AutoTokenizer

file_path = "/home/seungone/T0_continual_learning/ct0/data/sequential/train.wiki_auto.continual1000_preprocessed.json"

tokenizer = AutoTokenizer.from_pretrained('bigscience/T0_3B')

data_list = []
with open(file_path,'r') as f:
    for row in f:
        data = json.loads(row)
        print(data)
        print(str(data['en1']))
        print(str(data['en2']))
        print()
        print(tokenizer.batch_encode_plus([data['en1']], max_length=512,padding=True,truncation=True,return_tensors='pt'))
        print()
        print(tokenizer.batch_encode_plus([data['en2']], max_length=512,padding=True,truncation=True,return_tensors='pt'))
        break
        data_list.append(data)

    