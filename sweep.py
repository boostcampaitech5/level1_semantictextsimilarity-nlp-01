import torch
import pandas as pd
import pytorch_lightning as pl
from models.model import Model
from args import parse_arguments
from dataloader.dataloader import Dataloader
import wandb
from pytorch_lightning.loggers import WandbLogger
pl.seed_everything(420)

if __name__ == '__main__':
    args = parse_arguments()

    # Sweep 통해 실행될 학습 코드 생성 
    def sweep_train(config=None):
        wandb.init(config=config, entity=args.entity, project=args.project_name)
        config = wandb.config

        # Dataloader 정의
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.dataset_commit_hash)
        
        # model 정의
        model = Model(args.model_name, config.lr)

        wandb_logger = WandbLogger(project=args.project_name)

        # 가속기 설정
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        # Trainer 정의
        trainer = pl.Trainer(accelerator=accelerator, 
                            devices=1, 
                            max_epochs=config.epochs, 
                            log_every_n_steps=1,
                            logger=wandb_logger,
                            precision=16)

        # 학습
        trainer.fit(model=model, datamodule=dataloader)

        # 평가
        trainer.test(model=model, datamodule=dataloader)
    
    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=args.sweep_config,
        project=args.project_name
    )
    
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=args.sweep_count
    )