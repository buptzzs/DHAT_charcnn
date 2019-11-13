import json
import torchtext
from torchtext.data import Field, Dataset,Iterator, RawField, BucketIterator
from torchtext.vocab import Vocab, FastText
from tqdm import tqdm
import torch
import os
import numpy as np
from model import SimpleQANet
from dataset import DocumentTextField

import random
import argparse


class config:
    # *****************Dataset***********
    hidden = 50
    embedding_dim = 300
    dropout_prob = 0.2
    dropout = 0.2

    use_mentions = True
    # san
    memory_dropout = 0.4
    memory_type = 2 # 0. san 1. avg 2. last
    san_type = 3 # 0: baseline, use two self-att layer 1: use self-att first, then use bilinear-att 2: concate first, the use bilinear-att 3. use two bilinear-att layer
    steps = 5
    seed = 1023
    
    model_path = './train_log/models/san_type3_lr0.0005_steps5_mtype2_bs4/best.pt'
    


def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(config.seed)

class WikihopTestset(Dataset):

    fix_doc_field = {
        'batch_first':True,
        'lower':True,
        'memory_size':None,
        'sequential':True,
        'fix': 256
    }
    doc_field = {
        'batch_first':True,
        'lower':True,
        'memory_size':None,
        'sequential':True,
        'fix': None
    }
    
    vocab_path = '/home/zzs/data/qangaroo_v1.1/wikihop/train_val_vocab.pt'

    def __init__(self, path,  **kwargs):
        make_example = torchtext.data.example.Example.fromdict
        json_data = json.load(open(path))
        vocab = torch.load(self.vocab_path)
        examples = []
        
        fix_doc_field = DocumentTextField(**self.fix_doc_field)
        doc_field = DocumentTextField(**self.doc_field)
        doc_field_q = DocumentTextField(**self.doc_field)
        
        fix_doc_field.vocab = vocab
        doc_field.vocab = vocab
        doc_field_q.vocab = vocab

        fields = {
            'candidates': ('candidates',doc_field),
            'supports': ('supports',fix_doc_field),
            'query': ('query', doc_field_q),
            'id': ('id', RawField()),
        }        


        for d in tqdm(json_data):
            d = self.preprocess(d)
            example = make_example(d, fields)
            mentions = self.add_mention(example)
            example.mentions = mentions
            examples.append(example)
        fields['mentions'] = ('mentions',RawField())

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        super(WikihopTestset, self).__init__(examples, fields, **kwargs)

    def preprocess(self, item):
        item['query'] = item['query'].replace('_',' ')
        return item

    def add_mention(self, example):
        candidates = example.candidates
        supports = example.supports
        all_mentions = []
        for candidate in candidates:
            mentions = []
            c = ' '.join(candidate)
            for idx, support in enumerate(supports):
                for i in range(len(support)):
                    token = support[i]
                    if token == candidate[0]:
                        s = ' '.join(support[i:i+len(candidate)])
                        if s == c:
                            mentions.append([idx, i, i+len(candidate)])
            all_mentions.append(mentions)
        return all_mentions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str)
    parser.add_argument('out_path',type=str)

    args = parser.parse_args()
    dev_path = args.path
    dev_data = json.load(open(dev_path))
    id2data = {}
    for item in dev_data:
        id2data[item['id']] = item

    testset = WikihopTestset(dev_path)
    device = torch.device('cuda:0')
    test_iter = BucketIterator(testset, 4 ,sort_key=lambda x: len(x.supports), sort_within_batch=True, device=device)
    vocab = testset.fields['supports'].vocab
    net = SimpleQANet(config, vocab.vectors,  device)
    state_dict = torch.load(config.model_path)
    net.load_state_dict(state_dict['model'])

    net.eval()
    preds = {}
    for data in tqdm(test_iter):
        with torch.no_grad():
            score = net(data, return_label=False)
        pred = torch.argmax(score,dim=-1)
        for i in range(data.candidates.shape[0]):
            preds[data.id[i]] = pred[i].item()    

    test_pred = {}
    for id, idx in preds.items():
        test_pred[id] = id2data[id]['candidates'][idx]

    json.dump(test_pred, open(args.out_path,'w'))
