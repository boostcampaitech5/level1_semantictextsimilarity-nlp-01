import torch
import pytorch_lightning as pl
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoModelForSequenceClassification
import wandb
from wandb import AlertLevel
from LR_scheduler import CosineAnnealingWarmupRestarts
pl.seed_everything(420)

class Model(pl.LightningModule):
    def __init__(self, model_name, lr, loss_function='L1Loss'):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.loss_function = loss_function

        # 사용할 모델을 호출합니다.
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            num_labels=1,
        )
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = getattr(torch.nn, self.loss_function)()

        # val logit 값 출력 
        self.validation_predictions = []

    def forward(self, x):
        x = self.plm(x)['logits']
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", pearson_corrcoef(logits.squeeze(), y.squeeze()))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pearson = pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("test_pearson", pearson)

        wandb.alert(
		    title="test_step",
		    level=AlertLevel.INFO,
		    text=f'test_pearson : {pearson}'
		)

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
        # scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, 
        #                                           cycle_mult=1.0, max_lr=1e-5, min_lr=1e-6, 
        #                                           warmup_steps=50, gamma=0.5) 
        return [optimizer]
    # {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step"  # 스텝별로 스케줄러를 업데이트하도록 설정
    #         }
    #     }

    def on_train_epoch_start(self):
        # 에폭 시작 시 학습률을 로깅합니다.
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr)
