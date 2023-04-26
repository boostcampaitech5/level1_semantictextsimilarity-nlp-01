import torch
import pytorch_lightning as pl
from torchmetrics import PearsonCorrCoef, F1Score
from transformers import AutoModelForSequenceClassification

import wandb
from wandb import AlertLevel
from LR_scheduler import CosineAnnealingWarmupRestarts


class Model(pl.LightningModule):
    def __init__(self,
                 model_name: str,
                 optimizer: str = 'AdamW',
                 lr: float = 1e-5,
                 loss_function: str = 'L1Loss',
                 beta: float = 0.2,
                 bce: bool = False,
                 is_schedule:bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.optimizer = optimizer
        self.lr = lr
        self.loss_function = loss_function
        self.beta = beta
        self.bce = bce
        self.is_schedule = is_schedule

        # 모델 호출
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            num_labels=1,
        )
        
        # 손실 함수(loss function) 정의
        if self.loss_function == 'SmoothL1Loss':
            self.loss_func = getattr(torch.nn, self.loss_function)(beta=self.beta)
        else: # L1Loss, MSELoss, etc.
            self.loss_func = getattr(torch.nn, self.loss_function)
            
        # 평가 지표(evaluation metric) 정의
        if self.bce:
            self.evaluation = F1Score(task='binary')
        else:
            self.evaluation = PearsonCorrCoef()

        # val logit 값 출력 
        self.validation_predictions = []

    def forward(self, x):
        """모든 호출에서 실행되는 연산."""
        
        x = self.plm(x)['logits']
        return x

    def training_step(self, batch, batch_idx):
        """training loss 계산.
        
        Args:
            batch: Dataloader의 output.
            batch_idx: batch의 인덱스.
        
        Returns:
            loss: loss값.
        """

        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """validation loss 및 피어슨 상관계수 계산.
        
        Args:
            batch: Dataloader의 output.
            batch_idx: batch의 인덱스.
        
        Returns:
            loss: loss값.
        """

        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        pearson = self.evaluation(logits.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        self.log('val_pearson', pearson)
        return loss

    def test_step(self, batch, batch_idx):
        """test 피어슨 상관계수 계산.
        
        Args:
            batch: Dataloader의 output.
            batch_idx: batch의 인덱스.
        """

        x, y = batch
        logits = self(x)
        pearson = self.evaluation(logits.squeeze(), y.squeeze())
        self.log('test_pearson', pearson)

        wandb.alert(title='test_step',
		            level=AlertLevel.INFO,
		            text=f'test_pearson : {pearson}')

    def predict_step(self, batch, batch_idx):
        """predict 데이터셋에 대한 예측값 생성.
        
        Args:
            batch: Dataloader의 output.
            batch_idx: batch의 인덱스.
        
        Returns:
            logits.squeeze(): batch에 대한 예측값.
        """

        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        """학습에 사용한 optimizer과 learning-rate scheduler 선택."""

        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.lr)
        if self.is_schedule:
            # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
            scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, 
                                                    cycle_mult=1.0, max_lr=1e-5, min_lr=1e-6, 
                                                    warmup_steps=50, gamma=0.5)
            configure =  {'optimizer': optimizer,
                          'lr_scheduler': {
                            'scheduler': scheduler,
                            'interval': 'step'
                            }
                        }
        else:
            configure = [optimizer]
        
        return configure

    def on_train_epoch_start(self):
        """에폭 시작 시 학습률 로깅."""

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr)
