import json
import torchtext
from torchtext.data import Field, Dataset,Iterator, RawField, BucketIterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, FastText
from tqdm import tqdm
import torch
import os
import numpy as np
from torchtext import data as textdata, vocab
from torchtext.data import Field
import copy
from collections import Counter
from typing import List

    
class conf:
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

    doc_char_field = {
        'batch_first':True,
        'lower':True,
        'keep_sent_len': 256,
        'keep_word_len': 16
    }

    root = '/home/zzs/data/qangaroo_v1.1/wikihop/'
    train_path = 'train.json'
    dev_path = 'dev.json'



class DocumentTextField(Field):

    def __init__(self, memory_size=None,fix=None, **kwargs):
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        super(DocumentTextField, self).__init__(tokenize=tokenizer, **kwargs)

        self.memory_size = memory_size
        self.fix = fix
        if self.fix is not None: #防止overwrite fix
            self.fix_length = fix
        print('field info:memory_size:{}, fix:{}'.format(memory_size, fix))

    def preprocess(self, x):
        if isinstance(x, list):
            ss =  super(DocumentTextField, self).preprocess(x)
            return [super(DocumentTextField, self).preprocess(s) for s in ss]
        else:
            return super(DocumentTextField, self).preprocess(x)

    def pad(self, minibatch):
        if isinstance(minibatch[0][0], list):
            self.fix_length = max(max(len(x) for x in ex) for ex in minibatch)
            if self.fix is not None:
                self.fix_length = min(self.fix_length, self.fix)

            if self.memory_size is None:
                memory_size = max(len(ex) for ex in minibatch)
            else:
                memory_size = self.memory_size
            padded = []
            for ex in minibatch:
                # sentences are indexed in reverse order and truncated to memory_size
                nex = ex[:memory_size]
                padded.append(
                    super(DocumentTextField, self).pad(nex)
                    + [[self.pad_token] * self.fix_length]
                    * (memory_size - len(nex))
                    )
            return padded
        else:
            return super(DocumentTextField, self).pad(minibatch)

    def numericalize(self, arr, device=None):
        if isinstance(arr[0][0], list):
            tmp = [
                super(DocumentTextField, self).numericalize(x, device=device).data
                for x in arr
            ]
            arr = torch.stack(tmp)
            if self.sequential:
                arr = arr.contiguous()
            return arr
        else:
            return super(DocumentTextField, self).numericalize(arr, device=device)


class CharField(Field):
    
    def __init__(
        self,
        pad_token='<pad>',
        unk_token='<unk>',
        batch_first=True,
        max_word_length=20,
        max_sentence_length=128,
        lower=True,
        **kwargs):
        super().__init__(
            sequential=True,  # Otherwise pad is set to None in textdata.Field
            batch_first=batch_first,
            use_vocab=True,
            pad_token=pad_token,
            unk_token=unk_token,
            lower=lower,
            **kwargs
        )
        self.max_word_length = max_word_length
        self.max_sentence_length = max_sentence_length
        
    def build_vocab(self, *args, **kwargs):
        sources = []
        for arg in args:
            if isinstance(arg, textdata.Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)

        counter = Counter()
        for data in sources:
            # data is the return value of preprocess().
            for para in data:
                if isinstance(para[0], list):
                    for sentence in para:
                        for word_chars in sentence:
                            counter.update(word_chars)
                else:
                    for word_chars in para:
                        counter.update(word_chars)                    
       
        specials = [self.unk_token, self.pad_token]

        self.vocab = vocab.Vocab(counter, specials=specials, **kwargs)
        
    def pad(self, minibatch: List[List[List[str]]]) -> List[List[List[str]]]:
        """
        Example of minibatch:
        ::
            [[['p', 'l', 'a', 'y', '<PAD>', '<PAD>'],
              ['t', 'h', 'a', 't', '<PAD>', '<PAD>'],
              ['t', 'r', 'a', 'c', 'k', '<PAD>'],
              ['o', 'n', '<PAD>', '<PAD>', '<PAD>', '<PAD>'],
              ['r', 'e', 'p', 'e', 'a', 't']
             ], ...
            ]
        """
        # If we change the same minibatch object then the underlying data
        # will get corrupted. Hence deep copy the minibatch object.
        
        if self.max_sentence_length is not None:
            max_sentence_length = self.max_sentence_length
        else:
            max_sentence_length = max(len(sent) for sent in minibatch)

        if self.max_word_length is not None:
            max_word_length = self.max_word_length            
        else:
            max_word_length = max(len(word) for sent in minibatch for word in sent)        

        padded_minibatch = []
        for sentence in minibatch:
            sentence_ch = []
            for word in sentence[:max_sentence_length]:
                sentence_ch.append(list(word))
            padded_minibatch.append(sentence_ch)

        for i, sentence in enumerate(padded_minibatch):
            for j, word in enumerate(sentence):
                char_padding = [self.pad_token] * (max_word_length - len(word))
                padded_minibatch[i][j].extend(char_padding)
                padded_minibatch[i][j] = padded_minibatch[i][j][:max_word_length]
            if len(sentence) < max_sentence_length:
                for _ in range(max_sentence_length - len(sentence)):
                    char_padding = [self.pad_token] * max_word_length
                    padded_minibatch[i].append(char_padding)

        return padded_minibatch

    def numericalize(self, batch, device=None):
        batch_char_ids = []
        for sentence in batch:
            sentence_char_ids = super().numericalize(sentence, device=device)
            batch_char_ids.append(sentence_char_ids)
        return torch.stack(batch_char_ids, dim=0)


class DocCharField(CharField):

    def __init__(
        self, 
        memory_size=None,
        max_word_length=20,
        max_sentence_length=128,
        keep_sent_len=128,
        keep_word_len=10,
        **kwargs
        ):
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        super(DocCharField, self).__init__(tokenize=tokenizer, **kwargs)
        self.memory_size = memory_size
        self.keep_sent_len = keep_sent_len
        self.keep_word_len = keep_word_len


    def preprocess(self, x):
        if isinstance(x, list):
            ss =  super(DocCharField, self).preprocess(x)
            return [super(DocCharField, self).preprocess(s) for s in ss]
        else:
            return super(DocCharField, self).preprocess(x)

    def pad(self, minibatch):
        if isinstance(minibatch[0][0], list):
            self.max_sentence_length = max(max(len(x) for x in ex) for ex in minibatch)
            if self.keep_sent_len is not None:
                self.max_sentence_length = min(self.keep_sent_len, self.max_sentence_length)
            self.max_word_length = max([len(word) for para in minibatch for sent in para for word in sent ])
            if self.keep_word_len is not None:
                self.max_word_length = min(self.keep_word_len, self.max_word_length)
                
            if self.memory_size is None:
                memory_size = max(len(ex) for ex in minibatch)
            else:
                memory_size = self.memory_size
            padded = []
            for ex in minibatch:
                # sentences are indexed in reverse order and truncated to memory_size
                nex = ex[:memory_size]
                padded.append(
                    super(DocCharField, self).pad(nex)
                )
                for _ in range(memory_size - len(nex)):
                    padded_sent = [[self.pad_token]*self.max_word_length for _ in range(self.max_sentence_length)]
                    padded[-1].append(padded_sent)
            return padded
        else:
            self.max_sentence_length = None
            self.max_word_length = max([len(word) for sent in minibatch for word in sent ])
            if self.keep_word_len is not None:
                self.max_word_length = min(self.keep_word_len, self.max_word_length)
            return super(DocCharField, self).pad(minibatch)

    def numericalize(self, arr, device=None):
        if isinstance(arr[0][0][0], list):
            tmp = [
                super(DocCharField, self).numericalize(x, device=device).data
                for x in arr
            ]
            arr = torch.stack(tmp)
            if self.sequential:
                arr = arr.contiguous()
            return arr
        else:
            return super(DocCharField, self).numericalize(arr, device=device)


class WikihopDataset(Dataset):

    def __init__(self, path, fields=None, **kwargs):
        make_example = torchtext.data.example.Example.fromdict
        json_data = json.load(open(path))

        examples = []

        self.doc_field = fields['supports'][0][1]
        self.doc_char_field = fields['supports'][1][1]

        pt_path = path.replace('.json','_ch.pt')
        if os.path.exists(pt_path):
            examples = torch.load(pt_path)
        else:
            if 'mentions' in fields:
                fields.pop('mentions')
            for d in tqdm(json_data):
                d = self.preprocess(d)
                example = make_example(d, fields)
                mentions = self.add_mention(example)
                if len(mentions[example.label])!=0:
                    example.mentions = mentions
                    examples.append(example)
            torch.save(examples, pt_path)
        fields['mentions'] = ('mentions',RawField())

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)
        #print(fields)
        super(WikihopDataset, self).__init__(examples, fields, **kwargs)

    def preprocess(self, item):
        answer = item['answer']
        candidates = item['candidates']
        label = candidates.index(answer)
        item['label'] = label
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
        #if len(all_mentions[example.label]) == 0:
        #    return None
        return all_mentions

    @classmethod
    def iters(self, batch_size=4, path='', train='train.json', val='dev.json', device=None):
        fix_doc_field = DocumentTextField(**conf.fix_doc_field)
        doc_field = DocumentTextField(**conf.doc_field)
        doc_field_q = DocumentTextField(**conf.doc_field)

        doc_char_field = DocCharField(**conf.doc_char_field)

        fields = {
            'candidates': [('candidates',doc_field), ('candidates_char', doc_char_field)],
            'supports': [('supports',fix_doc_field),('supports_char', doc_char_field)],
            'query': [('query', doc_field_q),('query_char', doc_char_field)],
            'label': ('label', Field(sequential=False, is_target=True, use_vocab=False)),
            'id': ('id', RawField()),
        }

        word_vocab = FastText()
        train, val = self.splits(path=path, train=train, validation=val, fields=fields)
        fix_doc_field.build_vocab(train, val, vectors=word_vocab)
        doc_char_field.build_vocab(train, val)

        doc_field.vocab = fix_doc_field.vocab
        doc_field_q.vocab = fix_doc_field.vocab
        vocab_path = os.path.join(path, 'train_val_vocab.pt')
        if not os.path.exists(vocab_path):
            torch.save(fix_doc_field.vocab, vocab_path)
            torch.save(doc_char_field.vocab, vocab_path.replace('.pt','_char.pt'))

        return BucketIterator.splits([train, val], batch_size=batch_size,sort_key=lambda x: len(x.supports), sort_within_batch=True, device=device)

    #def get_test_set()

if __name__ == '__main__':
    train_ds, val_ds = WikihopDataset.iters(batch_size=4, path=conf.root, train=conf.train_path, val=conf.dev_path)

    i = 0
    for batch in train_ds:
        print(batch)
        if i > 5:
            break
        i += 1

    for batch in val_ds:
        print(batch)
        if i > 5:
            break
        i += 1            

    

