import os
from args import parse_arguments

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from dataloader.dataloader import STSDataModule

pl.seed_everything(420)


def main(config):
    # 실험명 정의
    run_name = '{}_{}_{}_{}_{}'.format(
        config.arch['type'],
        config.dataloader['args']['batch_size'],
        config.trainer['epochs'],
        config.optimizer['args']['lr'],
        config.loss['type'],
    )
    
    # 예측 결과 모아두는 폴더 outputs 생성
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dataloader 정의
    dataloader = STSDataModule(
        model_name=config.arch['type'],
        batch_size=config.dataloader['args']['batch_size'],
        shuffle=config.dataloader['args']['shuffle'],
        dataset_commit_hash=config.dataloader['args']['dataset_commit_hash'],
        num_workers=config.dataloader['args']['num_workers'],
    )
    val_dataloader = STSDataModule(
        model_name=config.arch['type'],
        batch_size=config.dataloader['args']['batch_size'],
        shuffle=config.dataloader['args']['shuffle'],
        dataset_commit_hash=config.dataloader['args']['dataset_commit_hash'],
        num_workers=config.dataloader['args']['num_workers'],
        use_val_for_predict=True
    )

    # Acceleator 설정
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # Trainer 정의
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=config.n_gpu, 
                         max_epochs=config.trainer['epochs'], 
                         log_every_n_steps=1)
    
    # 추론
    if not config.dataloader['args']['k_fold']['enabled']:        
        # Model 정의
        load_dir = '{}{}.pt'.format(config.trainer['save_dir'], run_name)
        model = torch.load(load_dir)
        # 저장된 모델로 예측 진행
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비
        predictions = list(round(float(i), 3) for i in torch.cat(predictions))

    else: # k-fold 교차 검증
        results = []
        for k in range(config.dataloader['args']['k_fold']['k']):
            load_dir = '{}{}_{}-fold.pt'.format(config.trainer['save_dir'],
                                                run_name, k)
            model = torch.load(load_dir)
            
            # Inference part
            # 저장된 모델로 예측을 진행
            predictions = trainer.predict(model=model, datamodule=dataloader)

            # 예측된 결과를 형식에 맞게 반올림하여 준비
            predictions = list(round(float(i), 3) for i in torch.cat(predictions))
            results.append(predictions)
        
        predictions = np.round(np.mean(np.array(results), axis=0), 3)
    
     # 0 미만 또는 5 초과의 값을 clipping
    predictions = [min(5., max(0., x)) for x in predictions]
    
    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions

    output_path = f'./outputs/submission_{run_name}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output.to_csv(output_path, index=False)
    
    # valid dataset 예측값을 확인하여 사후 분석 수행을 위한 dev_output.csv 뽑아냄
    val_predict = trainer.predict(model=model, datamodule=val_dataloader)
    val_predict = list(round(float(i), 3) for i in torch.cat(val_predict))
    val_predict = [min(5., max(0., x)) for x in val_predict]
    
    dev_output = pd.read_csv('./data/dev.csv')
    dev_output['preds'] = val_predict
    dev_output['diff'] = dev_output['label'] - dev_output['preds']
    
    dev_output_path = f'outputs/dev_output_{run_name}.csv'
    os.makedirs(os.path.dirname(dev_output_path), exist_ok=True)
    dev_output.to_csv(dev_output_path, index=False)


if __name__ == '__main__':
    config = parse_arguments()
    main(config)
    
            

