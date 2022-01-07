import os
import json
import torch
from tqdm import tqdm
import pickle
from datasets import Dataset
import pdb

task_to_keys = {
    'clinc150': ("text", None),
    'snips': ("text", None),
    'banking77': ("text", None),
}

def load_intent_datasets(task_name, tokenizer=None, max_seq_length=512, split=False, ratio=0.25, check_data=False):
    print("Loading {}".format(task_name))
    if task_name == 'clinc150':
        datasets, label_list  = load_clinc150(split=split, ratio=ratio, check_data=check_data)
    elif task_name == 'banking77':
        datasets, label_list = load_banking77(split=split, ratio=ratio, check_data=check_data)
    elif task_name == 'snips':
        datasets, label_list = load_snips(split=split, ratio=ratio, check_data=check_data)


    return datasets, label_list 

def load_split_dict(dset_name, ratio):
    with open (f'./data/{dset_name}/fromours_ratio_{ratio}_raw2split.pkl','rb') as  f:
        split_label_dict = pickle.load(f)
    
    return split_label_dict


def load_clinc150(split=False, ratio=0.25, check_data=False):
    root_dir = './data/clinc150'
    print(f"split:{split}, ratio:{ratio}")
    source_path = os.path.join(root_dir, 'data_full.json')
    with open(source_path, encoding='utf-8') as f:
        docs = json.load(f)

    intents_val = []  # Every intents are in each dataset split.
    [intents_val.append(intent) for _, intent in docs['val']
        if intent not in intents_val]
    # add out-of-scope intent
    intents_val.append('oos')



    # domain classification
    # this file can be found on official github page of original clinc paper
    domain_path = os.path.join(root_dir, 'domains.json')
    with open(domain_path, encoding='utf-8') as f:
        domain_docs = json.load(f)
    # add out-of-scope intent
    domain_docs['oos'] = ['oos']

    intent2domain = {}
    intentLabel2names = {}

    for i, (domain, intents) in enumerate(domain_docs.items()):
        for intent in intents:
            intent_label = intents_val.index(intent)
            intent2domain[intent_label] = i
            intentLabel2names[intent_label] = (intent, domain)
    label2names_path = os.path.join(root_dir, 'intentLabel2names.json')
    with open(label2names_path, 'w') as f:
        json.dump(intentLabel2names, f)
    intent_label_numbers = sorted(intentLabel2names)
    label_list = [intentLabel2names[i] for i in intent_label_numbers]

    
    train_dataset = []
    dev_dataset = []
    test_ind_dataset = []
    test_ood_dataset = []

    if split == False:
        for mode in docs.keys():    
        #     is_augment = 'train' in mode and 'oos' not in mode

            for i, line in enumerate(tqdm(docs[mode], desc=f'{mode} set')):
                text, intent = line
                intent_label = int (intents_val.index(intent))
                
                example = {'text': text, 'label': intent_label}

                # print(example)
                if mode == 'train':
                    train_dataset.append(example)
                elif mode == 'val':
                    dev_dataset.append(example)
                elif mode == 'test':
                    test_ind_dataset.append(example)
                elif mode == 'oos_test':
                    test_ood_dataset.append(example)
    else:
        label_dict = load_split_dict('clinc150',ratio)
        n_ind_classes = round(15 * ratio) * 10 -1
        for mode in ['train','val','test']:    
        #     is_augment = 'train' in mode and 'oos' not in mode

            for i, line in enumerate(tqdm(docs[mode], desc=f'{mode} set')):
                text, intent = line
                intent_label = int (intents_val.index(intent))
                split_label = label_dict[intent_label]
                if check_data:
                    example = {'text': text, 'label': split_label, 'label_text': intent, 'indices': i}
                else:
                    example = {'text': text, 'label': split_label}

                # print(example)
                if mode == 'train':
                    if label_dict[intent_label] <= n_ind_classes:
                        train_dataset.append(example)

                elif mode == 'val':
                    if label_dict[intent_label] <= n_ind_classes:
                        dev_dataset.append(example)
                elif mode == 'test':
                    if label_dict[intent_label] <= n_ind_classes:
                        test_ind_dataset.append(example)
                    else :
                        test_ood_dataset.append(example)

    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
    return datasets, label_list

def load_banking77(split=False, ratio=0.25, check_data=False):
    category_path = os.path.join('data', 'banking77', 'categories.json')
    with open(category_path, 'r') as f:
        categories = json.load(f)
    
    datasets = {}
    assert split == True, "Must split banking77!"
    label_dict = load_split_dict('banking77',ratio)
    n_ind_classes = round(77 * ratio) -1
    train_dataset = []
    dev_dataset = []
    test_ind_dataset = []
    test_ood_dataset = []

    for mode in ['train', 'valid', 'test']:
        data_path = os.path.join('data', 'banking77', mode)
        label_path = os.path.join(data_path, 'label')
        text_path = os.path.join(data_path, 'seq.in')
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
            labels = [label.rstrip('\n') for label in labels]
        
        with open(text_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            texts = [text.rstrip('\n') for text in texts]

        for i, (text, label) in enumerate(zip(texts, labels)):
            if check_data:
                example = {'text': text, 'label': label_dict[categories.index(label)], 'label_text': label}
            else:
                example = {'text': text, 'label': label_dict[categories.index(label)]}
                
            if mode == 'train':
                if label_dict[categories.index(label)] <= n_ind_classes: 
                    train_dataset.append(example)
            elif mode == 'valid':
                if label_dict[categories.index(label)] <= n_ind_classes: 
                    dev_dataset.append(example)
            elif mode == 'test':
                if label_dict[categories.index(label)] <= n_ind_classes:
                    test_ind_dataset.append(example)
                else:
                    test_ood_dataset.append(example)
                       
    label_list = []
    label_name_path = os.path.join('data', 'banking77', f'labels_{ratio}.json')
    with open(label_name_path) as f:
        label_name_dict = json.load(f)
    for i in range(77):
        for k, v in label_name_dict.items():
            if v == i:
                label_list.append(k)
                break
    assert len(label_list) == 77, 'Missing labels of banking77'
    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
    return datasets, label_list


def load_snips(split=False, ratio=0.25, check_data=False):
    # category_path = os.path.join('data', 'banking77', 'categories.json')
    # with open(category_path, 'r') as f:
    categories = {}
    
    datasets = {}
    test_ood_datasets = []
    assert split == True, "Must split snips!"
    
    label_dict = load_split_dict('snips',ratio)
    n_ind_classes = round(7 * ratio) -1
    train_dataset = []
    dev_dataset = []
    test_ind_dataset = []
    test_ood_dataset = []

    for mode in ['train', 'valid', 'test']:
        data_path = os.path.join('data', 'snips', mode)
        label_path = os.path.join(data_path, 'label')
        text_path = os.path.join(data_path, 'seq.in')
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = f.readlines()
            labels = [label.rstrip('\n') for label in labels]
        if mode == 'train': # if first iteration
            label_idx = 0
            for label in labels:
                if label not in categories.keys():
                    categories[label] = label_idx
                    label_idx += 1
        with open(text_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
            texts = [text.rstrip('\n') for text in texts]

        for i, (text, label) in enumerate(zip(texts, labels)):
            if check_data:
                example = {'text': text, 'label': label_dict[categories[label]], 'label_text': label, 'indices': i}
            else:
                example = {'text': text, 'label': label_dict[categories[label]]}
            if mode == 'train':
                if label_dict[categories[label]] <= n_ind_classes:
                    train_dataset.append(example)
            elif mode == 'valid':
                if label_dict[categories[label]] <= n_ind_classes:
                    dev_dataset.append(example)
            elif mode == 'test':
                if label_dict[categories[label]] <= n_ind_classes:
                    test_ind_dataset.append(example)
                else:
                    test_ood_dataset.append(example)
                    
    label_list = []
    label_name_path = os.path.join('data', 'snips', f'labels_{ratio}.json')
    with open(label_name_path) as f:
        label_name_dict = json.load(f)
    for i in range(7):
        for k, v in label_name_dict.items():
            if v == i:
                label_list.append(k)
                break
    assert len(label_list) == 7, 'Missing labels of snips'

    datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
    return datasets, label_list


def collate_fn(batch, pad_token_id=50256):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [pad_token_id] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = []
    for f in batch:
        last_true_token_idx = f['input_ids'].index(pad_token_id) if pad_token_id in f['input_ids'] else len(f['input_ids'])
        input_mask.append([1.0] * last_true_token_idx + [0.0] * (max_len - last_true_token_idx))
    # input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    if 'indices' in batch[0]:
        indices = torch.LongTensor([f["indices"] for f in batch])
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
            "indices": indices,
        }
    else:
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
        }
    return outputs


def collate_fn_prefix(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["label"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if 'indices' in batch[0]:
        indices = torch.LongTensor([f["indices"] for f in batch])
        outputs = {
            "input_ids": input_ids,
            "label": labels,
            "indices": indices,
        }
    else:
        outputs = {
            "input_ids": input_ids,
            "label": labels,
        }
    return outputs


def collate_fn_debug(batch, pad_token_id=50256):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = []
    for f in batch:
        last_true_token_idx = f['attention_mask'].index(pad_token_id) if pad_token_id in f['attention_mask'] else len(f['input_ids'])
        input_mask.append([1.0] * last_true_token_idx + [0.0] * (max_len - last_true_token_idx))
    labels = [f["label"] for f in batch]
    sentences = [f['text'] for f in batch]
    label_txt = [f['label_text'] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if 'indices' in batch[0]:
        indices = torch.LongTensor([f["indices"] for f in batch])
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
            "indices": indices,
            'text': sentences,
            'label_text': label_txt,
        }
    else:
        outputs = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": labels,
            'text': sentences,
            'label_text': label_txt,
        }
    return outputs

def transfrom_data_dict(data_dict):
    new_data_dict = {}
    first_data = data_dict[0]
    for k, v in first_data.items():
        new_data_dict[k] = [v]
    for data in data_dict[1:]:
        for k, v in data.items():
            new_data_dict[k].append(v)
    return Dataset.from_dict(new_data_dict)

def preprocess_dataset_for_transformers(dataset):
    # datasets = {'train': train_dataset, 'validation': dev_dataset, 'test_ind': test_ind_dataset, 'test_ood': test_ood_dataset}
    
    datasets = {'train': transfrom_data_dict(dataset['train']), 'validation': transfrom_data_dict(dataset['validation']), 
                'test_ind': transfrom_data_dict(dataset['test_ind']), 'test_ood': transfrom_data_dict(dataset['test_ood'])}


    
    return datasets