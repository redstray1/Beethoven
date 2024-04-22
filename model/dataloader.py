import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from sklearn.model_selection import train_test_split
from augmentation import crop_mouth
import os
from pathlib import Path
import glob
import json
import cv2
import torchvision
import random

import torchvision.transforms.functional

class BEETDataset(Dataset):
    TRAIN_VAL_RANDOM_SEED = 42
    VAL_RATIO = 0.05

    def __init__(self, folders : List[str], train: bool = True, sp_model_prefix: str = None,
                 vocab_size: int = 2000, normalization_rule_name: str = 'nmt_nfkc_cf',
                 model_type: str = 'bpe', max_length: int = 128, max_frames : int = 150):
        
        self.dataset = []
        self.aligns = []

        aligns_files = []

        for folder in folders:

            cur_aligns = glob.glob(os.path.join(folder, 'alignments', '*.txt'))
            aligns_files.extend(cur_aligns)

            for align_dir in cur_aligns:
                
                with open(align_dir, 'r', encoding='utf8') as file:
                    text = file.readline()

                self.aligns.append(text)

                self.dataset.append({
                    'video_dir' : os.path.join(folder, 'samples', Path(align_dir).stem + '.mp4'),
                    'align_dir' : align_dir,
                    'align' : text,
                    'train' : True,
                    'flip' : True
                })
            
            

        if not os.path.isfile(sp_model_prefix + '.model'):
            SentencePieceTrainer.train(
                input=aligns_files, vocab_size=vocab_size,
                model_type=model_type, model_prefix=sp_model_prefix,
                normalization_rule_name=normalization_rule_name,
                pad_id=3,
            )

        self.sp_model = SentencePieceProcessor(model_file=sp_model_prefix + '.model')
        
        train_dataset, val_dataset, train_aligns, val_aligns = train_test_split(self.dataset, self.aligns, test_size=self.VAL_RATIO, random_state=self.TRAIN_VAL_RANDOM_SEED)

        self.dataset = train_dataset if train else val_dataset
        self.aligns = train_aligns if train else val_aligns
        
        if not train:
            for i in range(len(self.dataset)):
                self.dataset[i]['flip'] = False
                self.dataset[i]['train'] = False

        self.encoded_indices = self.sp_model.encode(self.aligns)

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = \
            self.sp_model.pad_id(), self.sp_model.unk_id(), \
            self.sp_model.bos_id(), self.sp_model.eos_id()
        
        self.max_length = max_length
        self.max_frames = max_frames
        self.vocab_size = self.sp_model.vocab_size()

        self.frame_height = 64
        self.frame_width = 128

    def text2ids(self, texts : Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.sp_model.encode(texts)
    
    def ids2text(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2
            ids = ids.cpu().tolist()

        return self.sp_model.decode(ids)
    
    def __len__(self):
        return len(self.aligns)
    
    def encode(self, text: str):
        return [self.bos_id] + self.text2ids(text) + [self.eos_id]
    


    def read_data(self, d : dict) -> Tuple[torch.Tensor, torch.Tensor, str]:
        flip = d['flip'] or False
        sub = d['align']

        if d['train']:
            flip = flip or random.random() > 0.5
        else:
            flip = False

        y = torch.tensor(self.encode(sub), dtype=torch.int32)
        
        transform_list = []
        if flip:
            transform_list.append(torchvision.transforms.functional.hflip)
        transform_list += [
            torchvision.transforms.Lambda(crop_mouth),
            torchvision.transforms.Normalize(mean=[0.7136, 0.4906, 0.3283],
                std=[0.113855171, 0.107828568, 0.0917060521]),
            torchvision.transforms.Resize(size=(64, 128))
        ]

        data_transform = torchvision.transforms.Compose(transform_list)
        x, _, _ = torchvision.io.read_video(d['video_dir'], output_format='TCHW')

        x = x / 255
        x = data_transform(x)

        return x, y, sub
        pass

    def __getitem__(self, item : int) -> Tuple[torch.Tensor, int]:
        x = torch.zeros(self.max_frames, 3, self.frame_height, self.frame_width)

        d = self.dataset[item]

        frames, align, sub = self.read_data(d)
        x[:, : frames.size(1) :, :] = frames

        length = frames.size(1)

        return x, align, length, item


