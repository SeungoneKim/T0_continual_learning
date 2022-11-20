import json
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets

map_to_train_dataset = {
    "phase1" : "./data/sequential/train.wiki_auto.continual1000_preprocessed.json",
    "phase2" : "./data/sequential/train.sequential.gigaword.from.wiki_auto.continual1000_preprocessed.json",
    "phase3" : "./data/sequential/train.sequential.haiku.from.wiki_auto-_gigaword.continual1000_preprocessed.json",
    "phase4" : "./data/sequential/train.sequential.covid_qa_deepset.from.wiki_auto-_gigaword-_haiku.continual1000_preprocessed.json",
    "phase5" : "./data/sequential/train.sequential.eli5.from.wiki_auto-_gigaword-_haiku-_covid_qa_deepset.continual1000_preprocessed.json",
    "phase6" : "./data/sequential/train.sequential.empathetic_dialogues.from.wiki_auto-_gigaword-_haiku-_covid_qa_deepset-_eli5.continual1000_preprocessed.json",
    "phase7" : "./data/sequential/train.sequential.eSNLI.from.wiki_auto-_gigaword-_haiku-_covid_qa_deepset-_eli5-_empathetic_dialogues.continual1000_preprocessed.json",
    "phase8" : "./data/sequential/train.sequential.twitter_top20.from.wiki_auto-_gigaword-_haiku-_covid_qa_deepset-_eli5-_empathetic_dialogues-_eSNLI.continual1000_preprocessed.json",
}
map_to_eval_dataset = {
    "wiki_auto": ["./data/wiki_auto/simplification_1.test_preprocessed.json","./data/wiki_auto/simplification_2.test_preprocessed.json"],
    "asset": ["./data/asset/simplification_1.validation_preprocessed.json","./data/asset/simplification_2.validation_preprocessed.json"],
    "gigaword": ["./data/gigaword/constrain_contain+make_a_title.test.json","./data/gigaword/constrain_end+make_a_title.test.json","./data/gigaword/constrain_start+make_a_title.test.json"],
    "haiku": ["./data/haiku/do_nothing.test.json"],
    "covid_qa": [""],
    "eli5": ["./data/eli5/generate_a_question_1.test_asks.json"],
    "emdg": ["./data/empathetic_dialogues/dialogue_with_emotion.test.json"],
    "esnli": ["./data/eSNLI/explain_why.test.json"],
    "twst": ["./data/twitter_top20/tweet_as+about.test.json"]
}


class CT0Dataset_train(Dataset):
    # phase 1 : 
    def __init__(self, tokenizer, source_len, target_len, phase, device):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        self.phase = phase
        self.device = device

        if self.phase == 'phase1':
            self.train_file_dir = map_to_train_dataset['phase1']
        elif self.phase == 'phase2':
            self.train_file_dir = map_to_train_dataset['phase2']
        elif self.phase == 'phase3':
            self.train_file_dir = map_to_train_dataset['phase3']
        elif self.phase == 'phase4':
            self.train_file_dir = map_to_train_dataset['phase4']
        elif self.phase == 'phase5':
            self.train_file_dir = map_to_train_dataset['phase5']
        elif self.phase == 'phase6':
            self.train_file_dir = map_to_train_dataset['phase6']
        elif self.phase == 'phase7':
            self.train_file_dir = map_to_train_dataset['phase7']
        elif self.phase == 'phase8':
            self.train_file_dir = map_to_train_dataset['phase8']
        
        #self.data_list = load_dataset("json",data_files=self.train_file_dir, split='train')
        self.data_list = []
        with open(self.train_file_dir,'r') as f:
            for row in f:
                d = json.loads(row)
                self.data_list.append(d)
        # self._indices=None
        # self.data_list.set_format(type="torch")
        
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        ctext = str(data['en1'])
        text = str(data['en2'])

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.target_len, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze().to(dtype=torch.long)
        source_mask = source['attention_mask'].squeeze().to(dtype=torch.long)
        
        target_ids = target['input_ids'].squeeze().to(dtype=torch.long)

        decoder_input_ids = target_ids[:-1].contiguous()
        lm_labels = target_ids[1:].clone().detach()
        lm_labels[target_ids[1:] == self.tokenizer.pad_token_id] = -100
        

        return {
            'input_ids': source_ids, 
            'attention_mask': source_mask, 
            'decoder_input_ids': decoder_input_ids,
            'label_ids': lm_labels
        }

class CT0Dataset_eval(Dataset):
    # phase 1 : 
    def __init__(self, tokenizer, source_len, target_len, eval_dataset_name, device):
        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len
        
        self.data_listset_name = eval_dataset_name
        self.device = device
        if eval_dataset_name == "wiki_auto":
            self.eval_file_dir = map_to_eval_dataset['wiki_auto']
            #self.data_list1 = load_dataset("json",data_files=self.eval_file_dir[0])['train']
            #self.data_list2 = load_dataset("json",data_files=self.eval_file_dir[1])['train']
            #self.data_list =  concatenate_datasets([self.data_list1,self.data_list2])
            self.data_list = []
            with open(self.eval_file_dir[0],'r') as f:
                for row in f:
                    d = json.loads(row)
                    self.data_list.append(d)
            with open(self.eval_file_dir[1],'r') as f:
                for row in f:
                    d = json.loads(row)
                    self.data_list.append(d)

        elif eval_dataset_name == "asset":
            self.eval_file_dir = map_to_eval_dataset['asset']
            # self.data_list1 = load_dataset("json",data_files=self.eval_file_dir[0])['train']
            # self.data_list2 = load_dataset("json",data_files=self.eval_file_dir[1])['train']
            # self.data_list =  concatenate_datasets([self.data_list1,self.data_list2])
            self.data_list = []
            with open(self.eval_file_dir[0],'r') as f:
                for row in f:
                    d = json.loads(row)
                    self.data_list.append(d)
            with open(self.eval_file_dir[1],'r') as f:
                for row in f:
                    d = json.loads(row)
                    self.data_list.append(d)
        elif eval_dataset_name == "gigaword":
            self.eval_file_dir = map_to_eval_dataset['gigaword']
            self.data_list1 = load_dataset("json",data_files=self.eval_file_dir[0])['train']
            self.data_list2 = load_dataset("json",data_files=self.eval_file_dir[1])['train']
            self.data_list3 = load_dataset("json",data_files=self.eval_file_dir[2])['train']
            self.data_list =  concatenate_datasets([self.data_list1,self.data_list2,self.data_list3])
        elif eval_dataset_name == "haiku":
            self.eval_file_dir = map_to_eval_dataset['haiku']
            self.data_list = load_dataset("json",data_files=self.eval_file_dir)['train']
        # elif self.phase == 'covid_qa':
        #     self.eval_file_dir = map_to_eval_dataset['covid_qa']
        #     self.data_list = load_dataset("json",data_files=self.eval_file_dir)
        elif eval_dataset_name == "eli5":
            self.eval_file_dir = map_to_eval_dataset['eli5']
            self.data_list = load_dataset("json",data_files=self.eval_file_dir)['train']
        elif eval_dataset_name == "emdb":
            self.eval_file_dir = map_to_eval_dataset['emdb']
            self.data_list = load_dataset("json",data_files=self.eval_file_dir)['train']
        elif eval_dataset_name == "eSNLI":
            self.eval_file_dir = map_to_eval_dataset['eSNLI']
            self.data_list = load_dataset("json",data_files=self.eval_file_dir)['train']
        elif eval_dataset_name == "twst":
            self.eval_file_dir = map_to_eval_dataset['twst']
            self.data_list = load_dataset("json",data_files=self.eval_file_dir)['train']
        # self._indices=None
        # self.data_list.set_format(type="torch")
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        ctext = str(data['en1'])
        if self.data_listset_name == 'asset':
            text = str(data['en2'][0])
        else:
            text = str(data['en2'])

        source = self.tokenizer.batch_encode_plus([ctext], max_length= self.source_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([text], max_length= self.target_len, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze().to(dtype=torch.long)
        source_mask = source['attention_mask'].squeeze().to(dtype=torch.long)
        
        target_ids = target['input_ids'].squeeze().to(dtype=torch.long)

        decoder_input_ids = target_ids[:-1].contiguous()
        lm_labels = target_ids[1:].clone().detach()
        lm_labels[target_ids[1:] == self.tokenizer.pad_token_id] = -100
        

        return {
            'input_ids': source_ids, 
            'attention_mask': source_mask, 
            'decoder_input_ids': decoder_input_ids,
            'label_ids': lm_labels
        }