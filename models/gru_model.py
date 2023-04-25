import torch
from typing import Optional, Tuple, List
from torch import Tensor
import pytorch_lightning as pl
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoModelForSequenceClassification, AutoConfig
from torch.nn import GRU, Linear, Tanh, Dropout
import wandb
from torchmetrics import PearsonCorrCoef, F1Score
from wandb import AlertLevel
from LR_scheduler import CosineAnnealingWarmupRestarts

class GRUModel(pl.LightningModule):
    """
    사전 학습된 모델과 Bi-directional GRU layer를 결합하여 시퀀스 분류 작업을 수행하는 PyTorch Lightning 모델

    Args:
        model_name (str): 사용할 사전 학습된 모델의 이름
        lr (float): optimizer의 학습률
        loss_function (str, optional): 사용할 손실 함수. 기본값은 'L1Loss'
        bce (bool, optional): 평가에 이진 교차 엔트로피(Binary Cross Entropy, BCE)를 사용할지 여부. 기본값은 False

    Attributes:
        plm (AutoModelForSequenceClassification): 시퀀스 분류를 위한 사전 학습 모델
        gru (GRU): Bi-directional GRU layer
        linear (Linear): 선형 변환 레이어
        tanh (Tanh): 하이퍼볼릭 탄젠트 활성화 함수
        dropout (Dropout): 정규화를 위한 드롭아웃 레이어
        evaluation (PearsonCorrCoef 또는 F1Score): 평가 지표로, 피어슨 상관 계수(Pearson Correlation Coefficient) 또는 F1 점수(F1 Score)를 사용합니다.
        loss_func (torch.nn.Module): 최적화에 사용되는 손실 함수
        validation_predictions (list): 훈련 중 검증 예측을 저장하기 위한 목록
    """
    def __init__(self, model_name: str, lr: float, loss_function: str ='L1Loss', bce: bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.bce = bce
        self.loss_function = loss_function

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
            self.loss_func = getattr(torch.nn, self.loss_function)(beta=0.1)
        else:
            self.loss_func = getattr(torch.nn, self.loss_function)()
        
        # 평가 지표(evaluation metric) 정의
        if self.bce:
            self.evaluation = F1Score(task='binary')
        else:
            self.evaluation = PearsonCorrCoef()

        # val logit 값 출력 
        self.validation_predictions = []

    def forward(self, x: Tensor) -> Tensor:
        """
        GRU model의 forward pass. Input Tensor가 pre-trained 모델을 통과해 bidirectional GRU layer와 linear layer를 통과

        Args:
            x (Tensor): 토크나이징된 sequence
        
        Returns:
            Tensor : 각 Input Sequence의 logits
        """
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

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        훈련 step에서 loss 계산

        Args:
            batch (Tuple[Tensor, Tensor]): 토큰화된 input sequence와 label을 포함하는 텐서 튜플 
            batch_idx (int): 현재 배치 index
        
        Returns:
            Tensor: loss 값 반환
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        검증 스텝에서 배치 데이터에 대한 loss와 val_pearson 계산

        Args:
            batch (Tuple[Tensor, Tensor]): 토큰화된 input sequence와 label을 포함하는 텐서 튜플 
            batch_idx (int): 현재 배치 index

        Returns:
            Tensor: loss 값 반환
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        pearson = self.evaluation(logits.squeeze(), y.squeeze())
        self.log('val_loss', loss)
        self.log('val_pearson', pearson)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        테스트 스텝에서 배치 데이터에 대한 loss와 val_pearson 계산, val data와 동일하게 사용

        Args:
            batch (Tuple[Tensor, Tensor]): 토큰화된 input sequence와 label을 포함하는 텐서 튜플 
            batch_idx (int): 현재 배치 index
        """
        x, y = batch
        logits = self(x)
        pearson = self.evaluation(logits.squeeze(), y.squeeze())
        self.log('test_pearson', pearson)

        wandb.alert(
		    title="test_step",
		    level=AlertLevel.INFO,
		    text=f'test_pearson : {pearson}'
		)

    def predict_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        입력 데이터에 대한 예측 라벨 생성

        Args:
            batch (Tensor): 토큰화된 sequence를 포함하는 입력 텐서
            batch_idx: 현재 배치 인덱스
        
        Returns:
            Tensor: 예측된 logit 값을 포함하는 텐서 반환 
        """
        x = batch
        logits = self(x)
        return logits.squeeze()

    def configure_optimizers(self) -> dict:
        """
        optimizer와 learning rate scheduler를 설정

        Returns:
            dict: opmizer와 lr_scheduler를 포함하는 dictionary 반환
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200, 
                                                  cycle_mult=1.0, max_lr=1e-5, min_lr=1e-6, 
                                                  warmup_steps=50, gamma=0.5) 
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step'
                }}

    def on_train_epoch_start(self) -> None:
        """
        에폭 시작 시, 현재 학습률 로깅
        """
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr)
