import torch
from torch.utils.data import Dataset, DataLoader
#import tfrecord
import PIL
from PIL import Image
import io
import os
import math
import random

def PositionalEncoding(pos, n_dim):
    #batch = len(pos)
    pe = torch.zeros(n_dim)
    for i in range(0, n_dim, 2):
        pe[i] = \
            math.sin(pos / (10000 ** ((2 * i) / n_dim)))
        pe[i + 1] = \
            math.cos(pos / (10000 ** ((2 * (i + 1)) / n_dim)))
    return pe#.to("cuda:0")

class ASAPDataset(Dataset):

    def __init__(self, data_dir, transform=None, train = True, positional_encoding=False):

        self.data_dir = data_dir
        self.transform = transform
        self.positional_encoding = positional_encoding
        self.train = train
        if train:
            self.label_path = os.path.join(data_dir, 'train.txt')
        else:
            self.label_path = os.path.join(data_dir, 'test.txt')


    def __len__(self):
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        return len(lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence_length = 5

        images = []
        labels = []
        img_paths = []

        with open(self.label_path, 'r') as f:
            lines = f.readlines()

        img_paths.append(lines[idx].split('---')[0])


        length = int(lines[idx].split('---')[2])

        frame_num = int(img_paths[0].split('/')[-1].split('.j')[0].split('Frame')[1])
        frame_num_sec = int(frame_num/30)

        frame_nums = []

        for i in range(sequence_length-1):
            tmp = random.randint(0, length)
            images.append(PIL.Image.open(f'{lines[idx].split("---")[0].split(".j")[0].split("Frame")[0]}Frame{30*(tmp)}.jpg'))
            frame_nums.append(tmp*30)

        images.append(PIL.Image.open(lines[idx].split('---')[0]))
        frame_nums.append(frame_num)

            #print(record['image/path'].tostring()) #.split('/')[-1].split('.')[0].split('Frame')[1])
        labels.append(int(lines[idx].split('---')[1].split(',')[0]))
        

        if len(images) > 1: # for videos
            transformed_images = []
            pe = torch.zeros((len(images), 512))
            for i in range(len(images)):
                transformed_images.append(self.transform(images[i]))
                #pe[i] = PositionalEncoding(int(img_paths[0].split('/')[-1].split('.j')[0].split('Frame')[1]), 2048)
                if self.positional_encoding:
                    pe[i] = PositionalEncoding(int(frame_nums[i]), 512)
            return torch.stack(transformed_images), labels[0], pe, img_paths[0]
        else:
            pe = torch.zeros(512)
            if self.positional_encoding:
                pe = PositionalEncoding(int(img_paths[0].split('/')[-1].split('.j')[0].split('Frame')[1]), 2048)
            #return self.transform(images[0]), labels[0][0], labels[0][1], labels[0][2], pe
            return self.transform(images[0]), labels[0], pe , img_paths[0] 
