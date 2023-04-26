import torch
import pytorch_lightning as pl
from models.model import Model
from args import parse_arguments
from dataloader.dataloader import STSDataModule
import wandb
from pytorch_lightning.loggers import WandbLogger
from typing import Any


def main(config: Any) -> None:
    """
    WandB sweep 설정을 통한 하이퍼파라미터 튜닝 수행, config.json 파일로 설정 변경 가능

    Args:
        config: 사용자 정의 설정파일, sweep 조절 인자와 그렇지 않은 인자가 모두 포함됨
    """

    # Sweep 통해 실행될 학습 코드 생성 
    def sweep_train(config: Any = config) -> None:

        wandb.init(entity=config.wandb['entity'], # 기본값: 'salmons'
                   project=config.wandb['sweep_project_name'])
        sweep_config = wandb.config

        # 베이스라인 모델 혹은 GRU가 부가된 모델을 설정할 것인지 결정
        model_class = "gru_model.GRUModel" if config.arch['args']['gru_enabled']\
                    else "model.Model"
        module_name, class_name = model_class.split('.')
        model_module = __import__('models.' + module_name, fromlist=[class_name])
        ModelClass = getattr(model_module, class_name)

        # dataloader와 model을 정의합니다.
        dataloader = STSDataModule(
            model_name=config.arch['type'],
            batch_size=sweep_config['batch_size'],
            shuffle=config.dataloader['args']['shuffle'],
            dataset_commit_hash=config.dataloader['args']['dataset_commit_hash'],
            num_workers=config.dataloader['args']['num_workers'],
        )
        model = ModelClass(config.arch['type'],
                      sweep_config.lr,
                      config.loss['args']['beta'],
                      sweep_config.loss_function,
                      config.loss['args']['bce'],
                      config.lr_scheduler['is_schedule'])
        
        wandb_logger = WandbLogger()

        # 가속기 설정
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        # Trainer 정의
        trainer = pl.Trainer(accelerator=accelerator, 
                            devices=1, 
                            max_epochs=sweep_config.epochs, 
                            log_every_n_steps=1,
                            logger=wandb_logger,
                            precision=16)

        # 학습
        trainer.fit(model=model, datamodule=dataloader)

        # 평가
        trainer.test(model=model, datamodule=dataloader)
    
    # Sweep 생성
    sweep_id = wandb.sweep(
        sweep=config.sweep_config
    )
    
    wandb.agent(
        sweep_id=sweep_id,
        function=sweep_train,
        count=config.wandb['sweep_count']
    )

if __name__ == '__main__':
    config = parse_arguments()
    main(config)