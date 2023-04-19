import re

import pandas as pd
import pytorch_lightning as pl

from .dataset import *

from tqdm.auto import tqdm
from datasets import load_dataset
from typing import List, Tuple

from transformers import AutoTokenizer
from sklearn.model_selection import KFold
pl.seed_everything(420)


class KfoldDataloader(pl.LightningDataModule):
    """K-fold 교차 검증이 적용된 데이터 로더"""
    
    def __init__(self, 
                 model_name: str, 
                 batch_size: int, 
                 shuffle: bool, 
                 dataset_commit_hash: str,
                 k: int = 1,  # fold number
                 bce: bool = False,
                 num_folds: int = 5,
                 train_ratio: float = 0.8,
                 use_val_for_predict: bool = False):
        
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_commit_hash = dataset_commit_hash
        
        self.k = k
        self.bce = bce
        self.num_folds = num_folds
        self.train_ratio = train_ratio
        self.use_val_for_predict = use_val_for_predict
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       max_length=160)
        if not self.bce:
            self.target_columns = ['label']
        else: # binary cross entropy
            self.target_columns = ['binary_label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.columns = ['id', 'source', 'sentence_1', 'sentence_2', 
                        'label', 'binary-label']


    def tokenizing(self, dataframe: pd.DataFrame) -> List[str]:
        """주어진 문장을 토큰화.
        
        두 문장쌍을 [SEP] 토큰을 사이에 두고 이어붙이고 결과 문자열을 토큰화한 뒤
        리스트에 이어붙인다.

        Args:
            df: 다음의 열을 가진 데이터 프레임: 'source', 'sentence1', 'sentence2', 
                'label', and 'binary_label'.
                
        Returns:
            data: [SEP] 토큰을 사이에 두고 이어진 두 문장을 토큰화하여 얻은 토큰들의
                리스트.
        """
        
        data = []
        for idx, item in tqdm(dataframe.iterrows(), 
                              desc='tokenizing', 
                              total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[tc] for tc in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, 
                                     padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocess_text(self, text):
        """텍스트 전처리.
        
        - 특수문자 일부(. ? ! , ; : ^)만 남기고 제거하기.
        - 반복글자 줄이기.
        - 한글과 숫자만 남기기.
        
        Args: 
            text (str): 전처리 할 텍스트.
        
        Returns:
            text (str): 전처리 끝난 텍스트.
        """
        # <>사이의 이모티콘 제거
        text = re.sub('<.*?>', '', text)
        # 한글, 숫자, 특수문자 . ? ! , ; : ^만 남기기
        text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.,?!:;^]', '', text)
        # 특수문자 중복제거 
        text = re.sub(r'([.?!,:]){2,}', r'\1', text)
        # 양 공백 제거 
        text = text.strip()
        # 반복되는 글자를 2개로 줄여준다. 
        text = re.sub(r"(.)\1+", r"\1\1", text)
        return text

    def preprocessing(self, data: pd.DataFrame) \
            -> Tuple[List[str], List[float or int]]:
        """데이터 프레임 전처리.

        문장쌍을 전처리하고 이에 대한 정답 레이블을 추출하여 각각 입력과 타깃을 구성.

        Args:
            data: 데이터 프레임.

        Returns:
            inputs: [SEP] 토큰에 의해 이어진 두 문장을 토큰화하여 얻은 토큰들의
                리스트.
            targets: 정답 레이블을 포함한 리스트.
        """
        
        # 기본 텍스트 전처리
        data[self.text_columns] = data[self.text_columns].apply(
            lambda x: x.apply(self.preprocess_text))
        
        # 안쓰는 컬럼을 삭제합니다: 'id' column
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try: # target: 'label'
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)
        return inputs, targets

    def setup(self, stage: str = 'fit'):
        """데이터 준비.
        
        모델을 학습 및 검증 데이터에 연결하고, 데이터 전처리 및 변환에 필요한 객체들을
        초기화한다.
        
        Args:
            stage: 실행 스테이지. 'fit'이 기본값.
        """
        
        if stage == 'fit': # fit 스테이지
            # 데이터셋 로드
            # revision: tag name, or branch name, or commit hash
            total = load_dataset("Salmons/STS_Competition", 
                                 split='train', 
                                 column_names=self.columns, 
                                 revision=self.dataset_commit_hash)

            # pandas 형식으로 변환
            total_data = total.to_pandas().iloc[1:].reset_index(drop=True)\
                              .astype({'label': 'float', 'binary-label': 'float'})
            
            # 데이터 준비
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # 데이터셋을 num_folds번 fold
            kf = KFold(n_splits=self.num_folds, 
                       shuffle=self.shuffle)
            all_splits = [k for k in kf.split(total_dataset)]
            
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]
        else: # test / predict 스테이지
            test = load_dataset("Salmons/STS_Competition", 
                                split='validation', 
                                column_names=self.columns, 
                                revision=self.dataset_commit_hash)
            predict = load_dataset("Salmons/STS_Competition", 
                                   split='test', 
                                   column_names=self.columns, 
                                   revision=self.dataset_commit_hash)

            test_data = test.to_pandas().iloc[1:].reset_index(drop=True)\
                            .astype({'label':'float', 'binary-label':'float'})
            
            # val_dataset으로 predict 해서 결과물 보려고 구분하는 지점
            if self.use_val_for_predict:
                predict_data = test_data.copy()
            else:
                predict_data = predict.to_pandas().iloc[1:]\
                                      .reset_index(drop=True)[self.columns[:4]]

            # 평가데이터 준비
            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)

            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        """학습 데이터를 위한 데이터로더 객체 반환."""
        
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           shuffle=self.shuffle)

    def val_dataloader(self):
        """검증 데이터를 위한 데이터로더 객체 반환."""
        
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size)

    def test_dataloader(self):
        """평가 데이터를 위한 데이터로더 객체 반환."""
        
        return torch.utils.data.DataLoader(self.test_dataset, 
                                           batch_size=self.batch_size)

    def predict_dataloader(self):
        """예측 데이터를 위한 데이터로더 객체 반환."""
        
        return torch.utils.data.DataLoader(self.predict_dataset, 
                                           batch_size=self.batch_size)
