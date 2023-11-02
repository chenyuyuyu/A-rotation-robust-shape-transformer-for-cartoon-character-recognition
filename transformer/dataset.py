from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import torch
from py.config import C,M
import random


def collate(batch):
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
    )


def collate1(batch):
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        default_collate([b[2] for b in batch]),
        # default_collate([b[3] for b in batch]),
    )


class SCDataset(Dataset):
    def __init__(self, dataset, split, class_num, dist=False, transform=None):
        print(f"{split} dataset:", dataset)
        self.class_num = class_num
        txt_path = f"file_txt/{dataset}/{split}.txt"
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            word = line.split()
            imgs.append((word[0], int(word[1])))
        self.imgs = imgs
        self.transform = transform
        self.dist = dist

    def __len__(self):
        return len(self.imgs)

    def print_sample(self, idx: int = 0):
        fn, label = self.imgs[idx]
        print("file name:", fn, "\t label:", label)

    def __getitem__(self, idx):
        fn, label = self.imgs[idx]
        sc = np.load(fn)
        # [60,98,126,144,150,154]
        # if M.dim==60:
        #     sc=sc[:,:60]
        # elif M.dim==98:
        #     sc = sc[:, 60:158]
        # elif M.dim==126:
        #     sc = sc[:, 158:284]
        # elif M.dim==144:
        #     sc = sc[:, 284:428]
        # elif M.dim==150:
        #     sc = sc[:, 428:578]
        # elif M.dim==154:
        #     sc = sc[:, 578:]

        # num = random.randint(0, 14)  ##
        # if self.transform is not None and num != 0:
        #     # sc=self.transform(sc)
        #     tmp = sc[:num*10, :]
        #     sc[:150-num*10, :] = sc[num*10:, :]
        #     sc[150-num*10:, :] = tmp

        sc = torch.from_numpy(sc).float()  # (0,255)
        sc = sc.div(255)  # (0,1)
        sc = sc.sub_(0.5).div_(0.5)  # (x-mean)/std (-1,1)
        label = np.array(label)
        label = torch.from_numpy(label)
        # onthot=F.one_hot(label,num_classes=self.class_num) #torch.int64

        if self.dist is True: # only dist,no angle
            s_fn = fn.split("/")
            dist_fn = s_fn[0] + '/' + s_fn[1] + '_dist/' + s_fn[2] + '/' + s_fn[3][:-3] + 'txt'
            dist = np.loadtxt(dist_fn, delimiter='\t').reshape((-1, 1))
            # if self.transform is not None and num!=0:
            #     tmp = dist[:num*10, :]
            #     dist[:150-num*10, :] = dist[num*10:, :]
            #     dist[150-num*10:, :] = tmp

            dist = dist.astype(np.float32)
            max = np.max(dist)
            min = np.min(dist)
            dist_enc = (dist - min) / (max - min)
            dist_enc = torch.from_numpy(dist_enc)
            return sc, label, dist_enc
        # if self.dist is True: # dist+angle
        #     s_fn = fn.split("/")
        #     dist_fn = s_fn[0] + '/' + s_fn[1] + '_dist/' + s_fn[2] + '/' + s_fn[3][:-3] + 'txt'
        #     dist = np.loadtxt(dist_fn, delimiter='\t')
        #     # if self.transform is not None and num!=0:
        #     #     tmp = dist[:num*10]
        #     #     dist[:150-num*10] = dist[num*10:]
        #     #     dist[150-num*10:] = tmp
        #     dist = dist.astype(np.float32)
        #     max = np.max(dist)
        #     min = np.min(dist)
        #     dist_enc = (dist - min) / (max - min)
        #     dist_enc = torch.from_numpy(dist_enc)
        #
        #     angle_fn = s_fn[0] + '/' + s_fn[1] + '_angle/' + s_fn[2] + '/' + s_fn[3][:-3] + 'txt'
        #     angle = np.loadtxt(angle_fn, delimiter='\t')
        #     # if self.transform is not None and num!=0:
        #     #     tmp = angle[:num*10]
        #     #     angle[:150-num*10] = angle[num*10:]
        #     #     angle[150-num*10:] = tmp
        #     angle = angle.astype(np.float32)
        #     max = np.max(angle)
        #     min = np.min(angle)
        #     angle_enc = (angle - min) / (max - min)
        #     angle_enc = torch.from_numpy(angle_enc)
        #     return sc, label, dist_enc,angle_enc

        return sc, label
