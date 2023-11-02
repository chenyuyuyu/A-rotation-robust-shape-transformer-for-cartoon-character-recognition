#!/usr/bin/env python3
"""Train L-CNN
Usage:
    test.py [options] <yaml-config> <ckpt> <dataname>
    test.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file
   <ckpt>                          Path to ckpt
   <dataname>                      Dataset name

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-lr]
"""
# --->python test.py -d 0 -i <directory-to-storage-results> config/mpeg7.yaml <path-to-ckpt-file> mpeg7
import os
import time
import pprint
import random
import os.path as osp
import datetime
from skimage import io

import numpy as np
import torch
from docopt import docopt
from py.config import C,M
from py.lr_schedulers import init_lr_scheduler
from py.trainer import Trainer
from dataset import SCDataset,collate,collate1
from model import scTransformer
import torch.nn.functional as F



def build_model(cpu=False):
    model = scTransformer(
        num_classes=M.num_classes,
        heads=M.heads,
        # layers=M.layers,
        depth=M.depth,
        mlp_dim=M.mlp_dim,
        pool=M.pool,
        dropout=M.dropout,
        emb_dropout=M.emb_dropout,
        dist=M.dist,
        dim=M.dim
    )

    model = model.cuda()
    # model = DataParallel(model).cuda()

    if C.io.model_initialize_file:
        if cpu:
            checkpoint = torch.load(C.io.model_initialize_file, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(C.io.model_initialize_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        print('=> loading model from {}'.format(C.io.model_initialize_file))

    print("Finished constructing model!")
    return model


def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    C.io.model_initialize_file = args["<ckpt>"]
    C.io.dataname = args["<dataname>"]

    pprint.pprint(C, indent=4)
    bs = 100
    print("batch size: ", bs)
    print("data name: ", args["<dataname>"])

    # WARNING: L-CNN is still not deterministic
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")

    # 1. dataset
    if M.dist is False:
        kwargs = {
            # "batch_size": M.batch_size,
            "collate_fn": collate,
            "num_workers": C.io.num_workers,
            "pin_memory": True,
        }
    else:
        kwargs = {
            # "batch_size": M.batch_size,
            "collate_fn": collate1,
            "num_workers": C.io.num_workers,
            "pin_memory": True,
        }

    dataname = C.io.dataname
    test_loader = torch.utils.data.DataLoader(
        SCDataset(dataset=dataname, split="total", class_num=M.num_classes,dist=M.dist), batch_size=bs, **kwargs
    )
    data_size = len(test_loader) * bs

    # 2. model
    model = build_model()

    outdir = args["--identifier"]
    print("outdir:", outdir)
    file = open(outdir+'/result.txt', 'w+')
    file.write('[pred] ')
    file.write('[gt]\n')

    eval_time_=0
    correct=0
    model.eval()
    if M.dist is False:
        with torch.no_grad():
            for batch_idx, (sc, label) in enumerate(test_loader):
                input_dict = {
                    "sc": sc.cuda(),
                    "label": label.cuda()
                }
                # eval_t = time.time()
                pred = model(input_dict["sc"])
                pred = F.softmax(pred, dim=1)
                # eval_time_ += time.time() - eval_t
                pred = pred.max(1)[1]
                n_correct = pred.eq(input_dict["label"]).sum().item()
                correct += n_correct
                if batch_idx % 100 == 0:
                    print('  - accuracy: {accu:3.3f} '.format(
                        accu=100 * correct / (bs * (batch_idx + 1))))
                for i in range(pred.shape[0]):
                    file.write(str(pred[i].item()) + ' ')
                    file.write(str(label[i].item()) + '\n')
    else:
        with torch.no_grad():
            # for batch_idx, (sc, label,dist_enc,angle_enc) in enumerate(test_loader):
            for batch_idx, (sc, label,dist_enc) in enumerate(test_loader):
                input_dict = {
                    "sc": sc.cuda(),
                    "label": label.cuda(),
                    "dist_enc": dist_enc.cuda(),
                    # "angle_enc": angle_enc.cuda(),
                }
                eval_t = time.time()
                pred = model(input_dict)
                pred = F.softmax(pred, dim=1)
                eval_time_ += time.time() - eval_t
                pred = pred.max(1)[1]
                n_correct = pred.eq(input_dict["label"]).sum().item()
                correct += n_correct
                if batch_idx % 100 == 1:
                    print('  - accuracy: {accu:3.3f} '.format(
                        accu=100 * correct / (bs * (batch_idx + 1))))
                for i in range(pred.shape[0]):
                    file.write(str(pred[i].item()) + ' ')
                    file.write(str(label[i].item()) + '\n')
    accu=100 * correct/(bs*(batch_idx+1))
    file.write("accuracy: "+str(accu))
    file.close()
    print('  - Test accuracy: {accu:3.3f} '.format(
        accu=accu))
    print('image nums:',data_size)
    print('inference time:',eval_time_)





def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


if __name__ == "__main__":

    main()
