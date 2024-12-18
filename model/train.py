import time
from progressbar import *
import os, sys

from typing import List, Optional, Any

import random
import argparse
from collections import OrderedDict

from torchsummary import summary

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.models.decoder import ctc_decoder
import torch.nn.functional as F
from torcheval.metrics import WordErrorRate
from torcheval.metrics.functional import word_error_rate
from torchmetrics.functional import char_error_rate
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from model import BEETModel, LipNet



def training_epoch(beet : BEETModel, model : LipNet,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, pbar : ProgressBar,
                   writer : SummaryWriter, stats : dict,
                   epoch : int):
    
    device = next(model.parameters()).device

    for i_batch, sample_batched in enumerate(loader):
        pbar.update(i_batch + 1)
        niters += 1
        optimizer.zero_grad()
        x, y, lengths, y_lengths, idx = sample_batched

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()
        if torch.isnan(loss).any():
                print('Iteration with NaN loss')
                continue
        
        weight = torch.ones_like(loss_all)
        dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]

        logits.backward(dlogits)

        iter_loss = loss.item()
        writer.add_scalar('Train/Loss', iter_loss, niters)
        optimizer.step()
        stats['losses'][epoch] += iter_loss * x.size(0)

@torch.no_grad()
def validation_epoch(beet : BEETModel, model : LipNet,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, predictions : list, gt : list,
                   decoder : torchaudio.models.decoder.CTCDecoder,
                   stats : dict, epoch : int):
     
    device = next(model.parameters()).device

    def predict(logits, y, lengths, y_lengths, decoder, predictions, gt, n_show = 5):
        print('---------------------------------')

        n =min(n_show, logits.size(1))

        decoded = decoder(logits, lengths)

        predictions.append(decoded)

        cursor = 0
        for b in range(x.size(0)):
            y_str = beet.trainset.ids2text(y[b])
            gt.append(y_str)
            if b < n:
                 print('Test seq {} : {}; pred_{} : {}'.format(b + 1, y_str, 'beam', decoded[b]))
        

    for i_batch, sample_batched in enumerate(beet.valloader):
        x, y, lengths, y_lengths, idx = sample_batched
        x = x.to(device)

        logits = model(x)
        loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
        loss = loss_all.mean()

        if torch.isnan(loss).any():
            print('Iteration with NaN val loss')
            continue
        stats['losses_test'][epoch] += loss.item() * x.size(0)
        predict(logits, y, lengths, y_lengths, predictions=predictions, gt=gt, decoder=decoder, n_show=5)

          

          

def train(beet : BEETModel, model : LipNet, criterion : nn.Module,
          decoder : torchaudio.models.decoder.CTCDecoder, opt : dict):
    
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed(opt['seed'])
    np.random.seed(opt['seed'])
    random.seed(opt['seed'])
    torch.backends.cudnn.deterministic = True

    exp_name = '{}'.format(int(time.time()))

    device = next(model.parameters()).device

    log_dir = os.path.join('logs', exp_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    stats = {
        'losses': [0.] * opt['num_epochs'], 
        'losses_test': [0.] * opt['num_epochs'],
        'loss_ewma': 0.
    }
    niters = 0
    for epoch in range(opt['num_epochs']):
        optimizer = beet.optim(epoch)
        scheduler = beet.scheduler
        
        widgets = ['Epoch {}:'.format(epoch + 1), Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(beet.trainloader)).start()


        # one training epoch -----------------------------------------------
        model.train()
        for i_batch, sample_batched in enumerate(beet.trainloader):

            pbar.update(i_batch + 1)
            niters += 1

            optimizer.zero_grad()
            x, y, lengths, y_lengths, idx = sample_batched

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_all = criterion(F.log_softmax(logits, dim=-1), y, lengths, y_lengths)
            loss = loss_all.mean()

            if torch.isnan(loss).any():
                    print('Iteration with NaN loss')
                    continue
            
            weight = torch.ones_like(loss_all)
            dlogits = torch.autograd.grad(loss_all, logits, grad_outputs=weight)[0]

            logits.backward(dlogits)

            iter_loss = loss.item()
            writer.add_scalar('Train/Loss', iter_loss, niters)
            optimizer.step()
            stats['losses'][epoch] += iter_loss * x.size(0)



        stats['losses'][epoch] /= len(beet.trainset)
        pbar.finish()

        if epoch == 0:
             stats['loss_ewma'] = stats['losses'][epoch]
        else:
             stats['loss_ewma'] = stats['loss_ewma'] * 0.95 + stats['losses'][epoch] * 0.05

    predictions, gt = [], []
    print('Running Evaluation')

    # one validation epoch -----------------------------------------------
    model.eval()
    with torch.no_grad():
         validation_epoch(beet, model=model, optimizer=optimizer, criterion=criterion, loader=beet.valloader, stats=stats, decoder=decoder)


    if scheduler is not None:
        scheduler.step()

    #calculating losses
    stats['losses_test'][epoch] /= len(beet.valset)
    wer = word_error_rate(predictions, gt)
    cer = char_error_rate(predictions, gt)

    writer.add_scalar('Val/Loss', stats['losses_test'][epoch], niters)
    writer.add_scalar('Val/WER', wer, niters)
    writer.add_scalar('Val/CER', cer, niters)

    if epoch % opt.print_every == 0:
         print('Epoch{}: loss={:.5f}, avg={:.5f}, loss_val={:.5f}'.format(epoch+1, stats['losses'][epoch], stats['loss_ewma'], stats['losses_test'][epoch], min(stats['losses_test'][:epoch + 1])))
         print('WER:{:.4f}, CER: {:.4f}'.format(wer, cer))

        
        
    
