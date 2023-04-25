import os
import torch
import pytorch_lightning as pl

from args import parse_arguments
from models.model import Model
from dataloader.dataloader import STSDataModule

import wandb
from wandb import AlertLevel # logging level 지정시 사용
from pytorch_lightning.loggers import WandbLogger

pl.seed_everything(420)


def main(config):
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # WandB에서 사용될 실행 이름을 설정    
    run_name = '{}-{}-{}-{}-{}'.format(
        config.arch['type'],
        config.dataloader['args']['batch_size'],
        config.trainer['epochs'],
        config.optimizer['args']['lr'],
        config.loss['type'],
    )
    
    wandb.init(entity=config.wandb['entity'], # 기본값: 'salmons'
               project=config.wandb['project_name'],
               name=run_name)

    # PyTorch Lightning 의 logging과 WandB logger를 연결
    wandb_logger = WandbLogger()
    
    # 설정된 args를 실험의 hyperarams에 저장
    wandb_logger.log_hyperparams(config)

    # 모델 저장 위치 생성
    output_dir = config.trainer['save_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not config.dataloader['args']['k_fold']['enabled']:
        # Slack에 실험 시작 메시지를 보냄
        wandb.alert(title="start",
                    level=AlertLevel.INFO,
                    text=f'{run_name}')
      
        # Model 정의
        model = Model(config.arch['type'],
                      config.optimizer['args']['lr'],
                      config.loss['type'])
        
        # Dataloader 정의
        dataloader = STSDataModule(
            model_name=config.arch['type'],
            batch_size=config.dataloader['args']['batch_size'],
            shuffle=config.dataloader['args']['shuffle'],
            dataset_commit_hash=config.dataloader['args']['dataset_commit_hash'],
            num_workers=config.dataloader['args']['num_workers'],
        )
        
        # Trainer 정의
        trainer = pl.Trainer(accelerator=accelerator,
                             devices=config.n_gpu,
                             max_epochs=config.trainer['epochs'],
                             log_every_n_steps=1,
                             logger=wandb_logger,
                             precision=16)

        # 학습
        trainer.fit(model=model, datamodule=dataloader)
        # 추론
        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장
        torch.save(model, config.trainer['saved_dir'] + 'model.pt')
        
    else: # k-fold 교차 검증
        results = []
        for k in range(config.dataloader['args']['k_fold']['k']):
            # Slack에 실험 시작 메시지를 보냄
            wandb.alert(title="start",
                        level=AlertLevel.INFO,
                        text=f'{run_name}')

            # Model 정의
            model = Model(config.arch['type'],
                          config.optimizer['args']['lr'],
                          config.loss['type'])
            
            # Dataloader 정의
            dataloader = STSDataModule(
                model_name=config.arch['type'],
                batch_size=config.dataloader['args']['batch_size'],
                shuffle=config.dataloader['args']['shuffle'],
                dataset_commit_hash=config.dataloader['args']['dataset_commit_hash'],
                num_workers=config.dataloader['args']['num_workers'],
                k=k,
                bce=config.loss['args']['bce'],
                num_folds=config.dataloader['args']['k_fold']['k'],
            )

            # Trainer 정의
            trainer = pl.Trainer(accelerator=accelerator,
                                 devices=config.n_gpu,
                                 max_epochs=config.trainer['epochs'],
                                 log_every_n_steps=1,
                                 logger=wandb_logger,
                                 precision=16)
            
            
            trainer.fit(model=model, datamodule=dataloader)
            score = trainer.test(model=model, datamodule=dataloader)
            results.extend(score)
        
            torch.save(model, config.trainer['saved_dir'])
            
        # 모델의 평균 성능
        if config.loss['args']['bce']:
            result = [x['test_f1'] for x in results]
            score = sum(result) / config.dataloader['args']['k_fold']['k']
            print("K fold Test f1 score: ", score)
        else:
            result = [x['test_pearson'] for x in results]
            score = sum(result) / config.dataloader['args']['k_fold']['k']
            print("K fold Test pearson: ", score)
            
    wandb.finish()


if __name__ == '__main__':
    config = parse_arguments()
    main(config)
    