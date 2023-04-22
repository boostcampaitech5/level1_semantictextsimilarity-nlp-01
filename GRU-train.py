import torch
import pytorch_lightning as pl

from args import parse_arguments
from models.model import Model
from gru_model import GRUModel
from dataloader.dataloader import Dataloader
from dataloader.kfdataloader import KfoldDataloader

import wandb
from wandb import AlertLevel # logging level 지정시 사용
from pytorch_lightning.loggers import WandbLogger


if __name__ == '__main__':
    args = parse_arguments()

    model = GRUModel(args.model_name, args.learning_rate, args.loss_function)
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
    
    if not args.kfold:
        # slack에 실험 시작 메시지를 보냅니다.
        wandb.alert(title="start",
                    level=AlertLevel.INFO,
                    text=f'{run_name}')
      
        # dataloader와 model을 정의합니다.
        dataloader = Dataloader(model_name=args.model_name, 
                                batch_size=args.batch_size, 
                                shuffle=args.shuffle, 
                                dataset_commit_hash=args.dataset_commit_hash)

        # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
        trainer = pl.Trainer(accelerator=accelerator, 
                             devices=1, 
                             max_epochs=args.max_epoch, 
                             log_every_n_steps=1,
                             logger=wandb_logger,
                             precision=16)

        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        torch.save(model, args.saved_model_path)
        
    else:
        # num_folds는 fold의 개수, k는 k번째 fold datamodule
        results = []
        for k in range(args.num_folds):
            # slack에 실험 시작 메시지를 보냅니다.
            wandb.alert(title="start",
                        level=AlertLevel.INFO,
                        text=f'{run_name}')
          
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
                                 log_every_n_steps=1,
                                 logger=wandb_logger,
                                 precision=16)
            
            trainer.fit(model=model, datamodule=kfdataloader)
            score = trainer.test(model=model, datamodule=kfdataloader)
            results.extend(score)
        
            torch.save(model, 
                       f'{args.kfold_model_path}{args.model_name}-'\
                       f'{args.batch_size}-{args.max_epoch}-'\
                       f'{args.learning_rate}-{k}-of-{args.num_folds}-fold.pt')
            
        # 모델의 평균 성능
        if args.bce:
            result = [x['test_f1'] for x in results]
            score = sum(result) / args.num_folds
            print("K fold Test f1 score: ", score)
        else:
            result = [x['test_pearson'] for x in results]
            score = sum(result) / args.num_folds
            print("K fold Test pearson: ", score)
            
    wandb.finish()