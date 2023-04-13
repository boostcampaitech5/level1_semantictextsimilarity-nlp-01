import pandas as pd

import torch
import pytorch_lightning as pl

from args import parse_arguments
from dataloader.dataloader import Dataloader


if __name__ == '__main__':
    args = parse_arguments()

    # dataloader와 model을 정의합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, 
                            args.train_path, args.dev_path, args.test_path, 
                            args.predict_path)
    model = torch.load(args.saved_model_path)

    # gpu가 없으면 'gpus=0'을,
    # gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요.
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=1, 
                         max_epochs=args.max_epoch, 
                         log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = predictions
    output.to_csv('./data/output.csv', index=False)
