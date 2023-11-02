"""Train
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-lr]
"""
# --->python train.py -d 0 -i mpeg7 config/mpeg7.yaml
import os
import pprint
import random
import shutil
import os.path as osp
import datetime

import numpy as np
import torch
from docopt import docopt
import py
from py.config import C,M
from py.lr_schedulers import init_lr_scheduler
from py.trainer import Trainer
from dataset import SCDataset,collate,collate1
from model import scTransformer

def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(C.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    C.io.resume_from = outdir
    C.to_yaml(osp.join(outdir, "config.yaml"))
    return outdir

def build_model():
    model=scTransformer(
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

    # model = model.to(device)
    model=model.cuda()
    # model = DataParallel(model).cuda()
    if C.io.model_initialize_file:
        checkpoint = torch.load(C.io.model_initialize_file)
        model.load_state_dict(checkpoint["model_state_dict"], False)
        del checkpoint
        print('=> loading model from {}'.format(C.io.model_initialize_file))

    print("Finished constructing model!")
    return model

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"]
    C.update(C.from_yaml(filename=config_file))
    M.update(C.model)
    pprint.pprint(C, indent=4)
    resume_from = C.io.resume_from

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]
    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", torch.cuda.device_count(), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

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
    train_loader = torch.utils.data.DataLoader(
        SCDataset(dataset=dataname, split="train",class_num=M.num_classes,dist=M.dist), batch_size=M.batch_size, shuffle=True,
        drop_last=True, **kwargs
    )
    val_loader = torch.utils.data.DataLoader(
        SCDataset(dataset=dataname, split="val",class_num=M.num_classes,dist=M.dist), batch_size=M.eval_batch_size, **kwargs
    )
    epoch_size = len(train_loader)

    # 2. model
    model = build_model()

    # 3. optimizer
    if C.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=C.optim.lr,
            weight_decay=C.optim.weight_decay,
            amsgrad=C.optim.amsgrad,
            betas=(0.9, 0.98),
            eps=1e-09
        )
    else:
        raise NotImplementedError

    outdir = get_outdir(args["--identifier"])
    print("outdir:", outdir)

    iteration = 0
    epoch = 0
    best_mean_loss = 1e1000
    if resume_from:
        ckpt_pth = osp.join(resume_from, "checkpoint_lastest.pth.tar")
        checkpoint = torch.load(ckpt_pth)
        iteration = checkpoint["iteration"]
        epoch = iteration // epoch_size
        best_mean_loss = checkpoint["best_mean_loss"]
        print(f"loading {epoch}-th ckpt: {ckpt_pth}")

        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optim_state_dict"])

        lr_scheduler = init_lr_scheduler(
            optim, C.optim.lr_scheduler,
            stepsize=C.optim.lr_decay_epoch,
            max_epoch=C.optim.max_epoch,
            last_epoch=iteration // epoch_size
        )
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        del checkpoint

    else:
        lr_scheduler = init_lr_scheduler(
            optim,
            C.optim.lr_scheduler,
            stepsize=C.optim.lr_decay_epoch,
            max_epoch=C.optim.max_epoch
        )

    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optim,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=outdir,
        iteration=iteration,
        epoch=epoch,
        bml=best_mean_loss,
        dist=M.dist
    )


    trainer.train()

if __name__ == "__main__":
    # print(git_hash())
    main()