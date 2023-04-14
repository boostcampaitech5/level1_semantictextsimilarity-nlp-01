import torch
import pandas as pd
import pytorch_lightning as pl

from models.model import Model
from args import parse_arguments
from dataloader.dataloader import Dataloader

from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    args = parse_arguments()
    wandb_logger = WandbLogger(name=f'{args.model_name}#{args.batch_size}-{args.max_epoch}-{args.learning_rate}', project=args.project_name)
    
    # 설정된 args를 실험의 hyperarams에 저장합니다.
    wandb_logger.log_hyperparams(args)

    # dataloader와 model을 정의합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.dataset_commit_hash)
    model = Model(args.model_name, args.learning_rate)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=accelerator, 
                         devices=1, 
                         max_epochs=args.max_epoch, 
                         log_every_n_steps=1,
                         logger=wandb_logger)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, args.saved_model_path)
