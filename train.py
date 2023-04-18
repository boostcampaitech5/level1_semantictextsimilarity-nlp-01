import torch
import pandas as pd
import pytorch_lightning as pl
from models.model import Model
from args import parse_arguments
from dataloader.dataloader import Dataloader
import wandb
from pytorch_lightning.loggers import WandbLogger
from wandb import AlertLevel # logging level 지정시 사용
pl.seed_everything(420)

if __name__ == '__main__':
    args = parse_arguments()

    # wnadb에서 사용될 실행 이름을 설정합니다.
    run_name = f'{args.model_name}#{args.batch_size}-{args.max_epoch}-{args.learning_rate}'

    wandb.init(entity=args.entity, # config.default.json에 default값 'salmons'로 지정되어 있음
               project=args.project_name,
               name=run_name)

    # pytorch lightning 의 logging과 wandb logger를 연결합니다.
    wandb_logger = WandbLogger()
    
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
                         logger=wandb_logger,
                         precision=16)

    # slack에 실험 시작 메시지를 보냅니다.
    wandb.alert(title="start",
                level=AlertLevel.INFO,
                text=f'{run_name}')

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, args.saved_model_path)
