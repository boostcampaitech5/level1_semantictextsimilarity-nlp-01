import os

import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from args import parse_arguments
from dataloader.dataloader import Dataloader

pl.seed_everything(420)


if __name__ == '__main__':
    args = parse_arguments()
    
    # 예측 결과 모아두는 폴더 outputs 생성
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dataloader을 정의합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, 
                            args.shuffle, args.dataset_commit_hash)
    val_dataloader = Dataloader(args.model_name, args.batch_size, 
                                args.shuffle, args.dataset_commit_hash, 
                                use_val_for_predict=True)

    # gpu가 없으면 'gpus=0'을,
    # gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요.
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=1, 
                         max_epochs=args.max_epoch, 
                         log_every_n_steps=1)

    # Inference part
    if not args.kfold:
        run_name = f'snunlp_{args.batch_size}_{args.max_epoch}_{args.learning_rate}'
        # model을 정의합니다.
        model = torch.load(args.saved_model_path)
        # 저장된 모델로 예측을 진행합니다.
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        predictions = [min(5, max(0, x)) for x in predictions] # 5.1과 같은 값들을 5로, 0보다 작은 값들은 0으로 바꿔준다. 

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv(f'./outputs/output_{run_name}.csv', index=False)

        # valid dataset 예측값을 확인하여 사후 분석 수행을 위한 dev_output.csv 뽑아내기 
        val_predict = trainer.predict(model=model, datamodule=val_dataloader)
        val_predict = list(round(float(i), 1) for i in torch.cat(val_predict))
        val_predict = [min(5, max(0, x)) for x in val_predict]

        dev_output = pd.read_csv('./data/dev.csv')
        dev_output['preds'] = val_predict
        dev_output['diff'] = dev_output['label'] - dev_output['preds']
        dev_output.to_csv(f'./outputs/dev_output_{run_name}.csv', index=False, encoding='cp949')
    else:
        src, model = args.model_name.split('/')
        run_name = f'{args.model_name}-{args.batch_size}-{args.max_epoch}-{args.learning_rate}'
        results = []
        for k in range(args.num_folds):
            model = torch.load(f'{args.kfold_model_path}{args.model_name}-'\
                               f'{args.batch_size}-{args.max_epoch}-'\
                               f'{args.learning_rate}-{k}-fold.pt')
            # Inference part
            # 저장된 모델로 예측을 진행합니다.
            predictions = trainer.predict(model=model, datamodule=dataloader)

            # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
            predictions = list(round(float(i), 1) for i in torch.cat(predictions))
            results.append(predictions)
        
        predictions = np.mean(np.array(results), axis=0)

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('./data/sample_submission.csv')
        output['target'] = predictions
        
        folder_path = f'./data/output/{src}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        output.to_csv(f'./data/output/{run_name}.csv', index=False)
            

