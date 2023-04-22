import torch
import pytorch_lightning as pl
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.nn import GRU, Linear, Tanh, Dropout
import wandb
from torchmetrics import PearsonCorrCoef, F1Score
from wandb import AlertLevel
from LR_scheduler import CosineAnnealingWarmupRestarts
pl.seed_everything(420)

class GRUModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss_function='L1Loss', bce=False, optim='AdamW', beta=1.0):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.bce = bce
        self.loss_function = loss_function
        self.optim = optim
        self.beta = beta

        # Load the pre-trained model configuration
        config = AutoConfig.from_pretrained(model_name)
        config.update({"output_hidden_states": True, 
                       "num_labels" : 1})

        # 사용할 모델을 호출합니다.
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            config=config,
        )
        # Add a bidirectional GRU layer
        self.gru = GRU(input_size=2*self.plm.config.hidden_size, 
                       hidden_size=self.plm.config.hidden_size,
                       num_layers = 3,
                       dropout = 0.1,
                       batch_first=True, 
                       bidirectional=True)
        self.linear = Linear(in_features=2*self.plm.config.hidden_size, out_features=1)
        self.tanh = Tanh()
        self.dropout = Dropout(0.1)
        self.evaluation = PearsonCorrCoef()

        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        if self.loss_function=="SmoothL1Loss":
            self.loss_func = getattr(torch.nn, self.loss_function)(beta=self.beta)
        else:
            self.loss_func = getattr(torch.nn, self.loss_function)()
        
        if self.bce:
            self.evaluation = F1Score(task='binary')
        else:
            self.evaluation = PearsonCorrCoef()

        # val logit 값 출력 
        self.validation_predictions = []

    def forward(self, x):
        outputs = self.plm(x)['hidden_states'][-2:]
        # concat last two layers
        outputs = torch.cat(outputs, dim=-1)
        _, last_gru_state = self.gru(outputs)
        # Reshape to (num_layers, 2, batch_size, hidden_size)
        last_gru_state = last_gru_state.view(3, 2, -1, self.plm.config.hidden_size)  
        # Concatenate the last hidden state of forward and backward GRU
        last_hidden_state = torch.cat((last_gru_state[-1, 0], last_gru_state[-1, 1]), dim=-1)  
        x = self.tanh(last_hidden_state)
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        pearson = self.evaluation(logits.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        self.log('val_pearson', pearson)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pearson = self.evaluation(logits.squeeze(), y.squeeze())
        self.log('test_pearson', pearson)

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
        return [optimizer]

    def on_train_epoch_start(self):
        # 에폭 시작 시 학습률을 로깅합니다.
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr)
