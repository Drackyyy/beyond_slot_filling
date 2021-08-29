'''
Author: your name
Date: 2021-08-27 23:59:37
LastEditTime: 2021-08-28 23:47:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /simCSE/mfs/yangxiaocong/Workspace/beyond_slot_filling/main.py
'''
import torch
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from data import *
from model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--do_train', action="store_true", help='training phase or test phase. add this flag')

args = parser.parse_args()

seed_everything(0)
if __name__ == '__main__':
    dm = DialogueDataModule(max_len=256, data_ratio=1)
    model = TaggingModel()
    es = EarlyStopping(monitor='val_loss')
    logger = TensorBoardLogger(
        save_dir="experiments/",
        name = '100_percent_training_data')
    checkpoint_callback = ModelCheckpoint(
        verbose=True,
        monitor='val_loss',
        every_n_epochs=1
    )


    
    if args.do_train:
        trainer = Trainer(max_epochs = 20,precision = 16, gpus=[1],
                     logger = logger, callbacks=[es,checkpoint_callback],
                     log_gpu_memory="all",deterministic=True, auto_lr_find=True,auto_select_gpus=True)
        trainer.tune(model, dm)
        trainer.fit(model, dm)
    else:
        model = TaggingModel.load_from_checkpoint(
        checkpoint_path="/mfs/yangxiaocong/Workspace/beyond_slot_filling/experiments/100_percent_training_data/version_0/checkpoints/epoch=0-step=249.ckpt",
        hparams_file="/mfs/yangxiaocong/Workspace/beyond_slot_filling/experiments/100_percent_training_data/version_0/hparams.yaml",
        map_location=None,
)       
        trainer = Trainer(gpus=None)
        trainer.test(model, dm)