
import torch
import pytorch_lightning as pl

from args import parse_arguments
from models.model import Model
from dataloader.dataset import *
from dataloader.dataloader import *
from dataloader.kfdataloader import *

import wandb
from wandb import AlertLevel # logging level 지정시 사용
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    args = parse_arguments()

    model = Model(args.model_name, args.learning_rate)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # wnadb에서 사용될 실행 이름을 설정합니다.
    run_name = f'{args.model_name}#{args.batch_size}-{args.max_epoch}-{args.learning_rate}'

    wandb.init(entity=args.entity, # config.default.json에 default값 'salmons'로 지정되어 있음
               project=args.project_name,
               name=run_name)

    # pytorch lightning 의 logging과 wandb logger를 연결합니다.
    wandb_logger = WandbLogger()
    
    # 설정된 args를 실험의 hyperarams에 저장합니다.
    wandb_logger.log_hyperparams(args)
    
    # slack에 실험 시작 메시지를 보냅니다.
    wandb.alert(title="start",
                level=AlertLevel.INFO,
                text=f'{run_name}')
    
    if not args.kfold:
        # dataloader와 model을 정의합니다.
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, 
                                args.train_path, args.dev_path, args.test_path, 
                                args.predict_path)

        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(accelerator=accelerator, 
                             devices=1, 
                             max_epochs=args.max_epoch, 
                             log_every_n_steps=1)

        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, args.saved_model_path)
        
    else:
        # num_folds는 fold의 개수, k는 k번째 fold datamodule
        result = 0
        for k in range(args.num_folds):
            kfdataloader = KfoldDataloader(model_name=args.model_name, 
                                           batch_size=args.batch_size, 
                                           shuffle=args.shuffle, 
                                           dataset_commit_hash=args.dataset_commit_hash,
                                           k=k,
                                           bce=args.bce, 
                                           num_folds=args.num_folds)

            trainer = pl.Trainer(accelerator=accelerator, 
                                 devices=1, 
                                 max_epochs=args.max_epoch, 
                                 log_every_n_steps=1)
            trainer.fit(model=model, datamodule=kfdataloader)
            score = trainer.test(model=model, datamodule=kfdataloader)
            result += (score / args.num_folds)
        
            torch.save(model, 
                       f'{args.kfold_model_path}{args.model_name}-'\
                       f'{args.batch_size}-{args.max_epoch}-'\
                       f'{args.learning_rate}-{k}-fold.pt')