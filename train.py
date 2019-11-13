
import argparse

import os
import sys
import time
import collections
import json
import random
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter

from model import SimpleQANet
from dataset import WikihopDataset
import random

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class config:
    exp_dir = os.path.dirname(__file__)
    exp_name = os.path.basename(exp_dir)
    dir_path = os.path.abspath(exp_dir)
    log_dir = os.path.join(dir_path, 'train_log')
    log_model_dir = os.path.join(log_dir, 'models')

    # *****************Dataset***********
    root = '/home/zzs/data/qangaroo_v1.1/wikihop/'
    train_path = 'train.json'
    dev_path = 'dev.json'
    # *****************Model************
    hidden = 50
    embedding_dim = 300 + 100
    dropout_prob = 0.2
    dropout = 0.2
    
    # char cnn
    embed_dim = 8
    out_channels = 100
    kernel_size = 5


    use_mentions = True
    # san
    memory_dropout = 0.4
    memory_type = 2 # 0. san 1. avg 3
    san_type = 2 # 0: baseline, use two self-att layer 1: use self-att first, then use bilinear-att 2: concate first, the use bilinear-att 3. use two bilinear-att layer
    steps = 5


    # *****************Train************
    epochs = 50
    batch_size = 4
    seed = 1023

    n_total_epoch = 50
    base_lr = 5e-4
    checkpoint_interval = 1

    # ****************DEBUG**************
    debug = False
    test_iter = 10
    test_epoch = 2

    save_ckpt = False

    model_name = 'san_type{}_lr{}_steps{}_mtype{}_bs{}_maxmean'.format(san_type, base_lr, steps, memory_type, batch_size)



#torch.cuda.set_device(0)

def format_time(elapse):
    elapse = int(elapse)
    hour = elapse // 3600
    minute = elapse % 3600 // 60
    seconds = elapse % 60
    return "{:02d}:{:02d}:{:02d}".format(hour, minute, seconds)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1
        self.minibatch = 0

    def make_checkpoint(self):
        return {
            'epoch': self.epoch, 
            'minibatch': self.minibatch, 
            'step': self.step
        }

    def restore_checkpoint(self, clock_dict):
        self.epoch = clock_dict['epoch']
        self.minibatch = clock_dict['minibatch']
        self.step = clock_dict['step']        


class Session:
    def __init__(self, config, device, net = None, visualizer = None,optimizer=None):
        self.device = device
        self.config = config
        self.clock = TrainClock()

        self.net = net
        self.epoch = 0
        #self.net = nn.DataParallel(self.net.cuda(), device_ids=device_ids)
        #self.net.to(self.device)
        self.extra_info = collections.OrderedDict()
        self.opt = optimizer#.cuda()#to(self.device)

    def get_state_dict_on_cpu(self, obj):
        cpu_device = torch.device('cpu')
        state_dict = obj.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(cpu_device)
        return state_dict


    def save_ckpt(self, ckpt_name):
        ckpt_dict = {
            'epoch': self.epoch,
            'clock': self.clock.make_checkpoint(),
            'optimizer': self.opt.state_dict(),
            'model': self.get_state_dict_on_cpu(self.net)
        }
        model_dir = os.path.join(config.log_model_dir, config.model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(ckpt_dict, os.path.join(model_dir, ckpt_name+'.pt'))


    def load_ckpt(self, ckp_path):
        checkpoint = torch.load(ckp_path, dict)
        ckpt_dict = checkpoint
        self.net.load_state_dict(ckpt_dict['model'], strict=False)
        self.epoch = ckpt_dict['epoch']
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.opt.load_state_dict(checkpoint['optimizer'])

    # for logging
    def log_best_value(self, acc, epoch):
        if 'acc' not in self.extra_info:
            self.extra_info.setdefault('acc', 0.0)
            self.extra_info.setdefault('epoch', 0)
        if acc > self.extra_info['acc']:
            self.extra_info['acc'] = acc
            self.extra_info['epoch'] = epoch
            return True
        return False

    def tensorboards(self, *names):
        self.tb_loggers = [
            SummaryWriter(os.path.join(self.config.log_dir,d+config.model_name))
                for d in names
        ]
        return self.tb_loggers

    def train(self, data, learning_rate, idx, train_tb, loss_meter, acc_meter): ## minibatch
        self.net.train()
        score, label = self.net(data)
        loss = F.cross_entropy(score, label)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.25)
        self.opt.step()

        pred = torch.argmax(score,dim=-1)
        acc_num = pred.eq(label).sum().item()
        total = pred.size()[0]

        acc_meter.update(acc_num, total)
        loss_meter.update(loss.item(), 1)
        out_dict = {
            'loss': loss_meter.avg,
            'train_acc': acc_meter.avg
        }
        return out_dict

    def val(self, val_ds):
        self.net.eval()
        acc_avg = AverageMeter()
        for data in tqdm(val_ds):
            with torch.no_grad():
                score, label = self.net(data)
            pred = torch.argmax(score,dim=-1)
            acc_num = pred.eq(label).sum().item()
            total = pred.size()[0]
            acc_avg.update(acc_num, total)
        return acc_avg.avg

    def update_extra_info(self,epoch, idx, train_info):
        self.extra_info[(epoch, idx)] = train_info

    def adjust_learning_rate(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def get_learning_rate(self):
        for param_group in self.opt.param_groups:
            return param_group['lr']


def load_ckpt(model, ckpt_name):
    ckp_path = ckpt_name
    if ckpt_name.endswith('.link'):
        with open(ckpt_name) as f:
            ckp_path = f.read().strip()

    checkpoint = io.load(ckp_path, dict)
    pretrained_dict = checkpoint['model']
    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict)

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def seed_torch(seed=2018):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def init_emb(vocab, init="randn", num_special_toks=2):
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = 0
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            # std = 0.05 is based on the norm of average GloVE 100-dim word vectors
            if init == "randn":
                torch.nn.init.normal(emb_vectors[i], mean=0, std=0.05)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
        total_words += 1
    print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
        running_norm / num_non_zero, num_non_zero, total_words))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--device', default=0,type=int)
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    args = parser.parse_args()

    seed_torch(config.seed)
    logger.info(config.model_name)
    logger.info('load datasets')
    device = torch.device('cuda:{}'.format(args.device))

    train_ds, val_ds = WikihopDataset.iters(batch_size=config.batch_size, path=config.root, train=config.train_path, val=config.dev_path, device=device)
    vocab = train_ds.dataset.doc_field.vocab
    init_emb(vocab)

    char_embed_num =  len(train_ds.dataset.doc_char_field.vocab.itos)


    logger.info('load model')

    net = SimpleQANet(config, vocab.vectors,  char_embed_num, device)
    optimizer = optim.Adam(net.parameters(), lr = config.base_lr)
    sess = Session(config, device, net=net, optimizer=optimizer)

   # restore checkpoint
    if args.continue_path and os.path.exists(args.continue_path):
        logger.info('restore from checkpoint:{}'.format(args.continue_path))
        sess.load_ckpt(args.continue_path)

    train_tb, = sess.tensorboards("train")
    total_step = len(train_ds.dataset) // config.batch_size * config.n_total_epoch
    clock = sess.clock
    time_train_start = time.time()
    step_start = clock.step

    # accumulate dataset
    num_iter = len(train_ds) 

    logger.info('begin train')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_total_epoch)
    for epoch_data in range(config.n_total_epoch):
        if clock.epoch > config.n_total_epoch:
            break

        if clock.epoch >= config.test_epoch and config.debug:
            break

        time_iter_start = time.time()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for idx, mini_batch_data in enumerate(train_ds):
        #for idx in range(num_iter):
            #mini_batch_data = train_ds.next()
            if idx > config.test_iter and config.debug:
                break

            learning_rate = sess.get_learning_rate()
            train_tb.add_scalar('learning_rate', learning_rate, clock.step)

            tdata = time.time() - time_iter_start
            outputs_data_train = sess.train(mini_batch_data, learning_rate, clock.step, train_tb, loss_meter, acc_meter)

            time_train_passed = time.time() - time_train_start
            step_passed = clock.step - step_start
            eta = (total_step - clock.step) * 1.0 / max(step_passed, 1e-7) * time_train_passed
            time_iter_passed = time.time() - time_iter_start

            # print text info
            meta_info = list()
            meta_info.append('[{}/{}:{}/{}]'.format(clock.epoch, config.n_total_epoch, idx, num_iter))
            meta_info.append('lr:{:.5g}'.format(learning_rate))
            meta_info.append('{:.2g} b/s'.format(1. / time_iter_passed))
            meta_info.append('passed:{}'.format(format_time(time_train_passed)))
            meta_info.append('eta:{}'.format(format_time(eta)))
            meta_info.append('data_time:{:.2g}'.format(tdata / time_iter_passed))
            loss_info = ['{}:{:.4g}'.format(k, float(outputs_data_train[k])) for k in sorted(outputs_data_train.keys())]
            if idx % 100 == 0:
                info = [",".join(meta_info)] + loss_info
                logger.info(", ".join(info))

            # tensorboard info
            for k, v in outputs_data_train.items():
                train_tb.add_scalar(k, v, step_passed)

            time_iter_start = time.time()
            clock.tick()
        clock.tock()

        if clock.epoch % config.checkpoint_interval == 0 and config.save_ckpt:
            sess.save_ckpt('epoch-{}'.format(clock.epoch))
        if config.save_ckpt:
            sess.save_ckpt('latest')

        val_acc = sess.val(val_ds)
        logger.info('validation: epoch: {}, accuracy:{}'.format(clock.epoch, val_acc))
        better = sess.log_best_value(val_acc, clock.epoch)
        train_tb.add_scalar('val_acc', val_acc, clock.epoch)
        if better:
            sess.save_ckpt('best')
        result_path = '{}.json'.format(config.model_name)
        #if os.path.exists(result_path):
        #    result_path = result_path.replace('.json','_t2.json')
        json.dump(sess.extra_info, open(result_path,'w'))

        scheduler.step()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt, exit.")
        os._exit(1)
