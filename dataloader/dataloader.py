# dataloader.py
import re

import pandas as pd
import pytorch_lightning as pl

from tqdm.auto import tqdm
from typing import List, Tuple
from .dataset import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


class STSDataModule(pl.LightningDataModule):

    def __init__(self,
                 model_name,
                 batch_size,
                 shuffle,
                 dataset_commit_hash,
                 num_workers,
                 k: int = 1,  # fold number
                 bce: bool = False,
                 num_folds: int = 5,
                 k_fold_enabled: bool = False,
                 use_val_for_predict: bool = False):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_commit_hash = dataset_commit_hash

        self.num_workers = num_workers

        self.k = k
        self.bce = bce
        self.num_folds = num_folds
        self.k_fold_enabled = k_fold_enabled

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       max_length=160)

        if not self.bce:
            self.target_columns = ['label']
        else:
            self.target_columns = ['binary_label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.columns = ['id', 'source', 'sentence_1', 'sentence_2',
                        'label', 'binary-label']

        self.use_val_for_predict = use_val_for_predict

    def preprocess_text(self, text: str):
        """텍스트 전처리.

        - 특수문자 일부(. ? ! , ; : ^)만 남기고 제거하기.
        - 반복글자 줄이기.
        - 한글과 숫자만 남기기.

        Args:
            text (str): 전처리 할 텍스트

        Returns:
            text (str): 전처리 끝난 텍스트
        """

        # <>사이의 이모티콘 제거
        text = re.sub('<.*?>', '', text)
        # 한글, 영어, 숫자, 특수문자 ? ; : ^만 남김
        text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.,?!:;^]', '', text)
        # ㅎ, ㅋ, ㅠ 제거
        text = re.sub('[ㅋㅎㅠ]+', '', text)
        # 특수문자 중복제거
        text = re.sub(r'([.?!,:]){2,}', r'\1', text)
        # 양 공백 제거
        text = text.strip()
        # 반복되는 글자들을 2개로 줄임
        text = re.sub(r"(.)\1+", r"\1\1", text)
        return text

    def tokenizing(self, dataframe: pd.DataFrame) -> List[str]:
        """주어진 문장을 토큰화.

        두 문장쌍을 [SEP] 토큰을 사이에 두고 이어붙이고 결과 문자열을 토큰화한 뒤
        리스트에 이어붙임.

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
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리
            text = '[SEP]'.join([item[text] for text in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True,
                                     padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self,
                      data: pd.DataFrame,
                      is_train: bool = False)\
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

        # 사용하지 않는 컬럼 삭제
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴
        try:
            targets = data[self.target_columns].values.tolist()
        except KeyError:
            targets = []

        # 텍스트 데이터 전처리
        inputs = self.tokenizing(data)
        return inputs, targets

    def setup(self, stage: str = 'fit'):
        """데이터 준비.

        모델을 학습 및 검증 데이터에 연결하고, 데이터 전처리 및 변환에 필요한 객체들을
        초기화한다.

        Args:
            stage: 실행 스테이지. 'fit'이 기본값.
        """

        # train stage
        if stage == 'fit':

            # 데이터셋 로드
            train = load_dataset("Salmons/STS_Competition",
                                 split='train',
                                 column_names=self.columns,
                                 revision=self.dataset_commit_hash)

            if not self.k_fold_enabled:
                valid = load_dataset("Salmons/STS_Competition",
                                     split='validation',
                                     column_names=self.columns,
                                     revision=self.dataset_commit_hash)

                # pandas 형식으로 변환
                train_data = train.to_pandas().iloc[1:].reset_index(drop=True)\
                                  .astype({'label': 'float',
                                           'binary-label': 'float'})
                val_data = valid.to_pandas().iloc[1:].reset_index(drop=True)\
                                .astype({'label': 'float',
                                         'binary-label': 'float'})

                # train 데이터 및 val 데이터 준비
                train_inputs, train_targets = self.preprocessing(train_data)
                val_inputs, val_targets = self.preprocessing(val_data)

                # train 데이터셋만 shuffle 적용
                # 필요에 따라 val, test 데이터셋에도 shuffle 적용 가능
                self.train_dataset = Dataset(train_inputs, train_targets)
                self.val_dataset = Dataset(val_inputs, val_targets)

            # k-fold dataloader
            else:
                total_data = train.to_pandas().iloc[1:].reset_index(drop=True)\
                                  .astype({'label': 'float',
                                           'binary-label': 'float'})

                # 데이터 준비
                total_input, total_targets = self.preprocessing(total_data)
                total_dataset = Dataset(total_input, total_targets)

                # 데이터셋을 num_folds번 fold
                kf = KFold(n_splits=self.num_folds,
                           shuffle=self.shuffle)
                all_splits = [k for k in kf.split(total_dataset)]

                # k번째 fold 된 데이터셋의 index 선택
                train_indexes, val_indexes = all_splits[self.k]
                train_indexes, val_indexes = \
                    train_indexes.tolist(), val_indexes.tolist()

                # fold한 index에 따라 train, val 데이터셋 분할
                self.train_dataset = [total_dataset[x] for x in train_indexes]
                self.val_dataset = [total_dataset[x] for x in val_indexes]

        # test stage
        else:
            test = load_dataset("Salmons/STS_Competition",
                                split='validation',
                                column_names=self.columns,
                                revision=self.dataset_commit_hash)
            predict = load_dataset("Salmons/STS_Competition",
                                   split='test',
                                   column_names=self.columns,
                                   revision=self.dataset_commit_hash)

            test_data = test.to_pandas().iloc[1:].reset_index(drop=True)\
                            .astype({'label': 'float',
                                     'binary-label': 'float'})

            # predict할 데이터셋 선택 (val / test)
            if self.use_val_for_predict:
                predict_data = test_data.copy()
            else:
                predict_data = predict.to_pandas().iloc[1:]\
                                      .reset_index(drop=True)[self.columns[:4]]

            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)

            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        """학습 데이터를 위한 데이터로더 객체 반환."""

        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle)

    def val_dataloader(self):
        """검증 데이터를 위한 데이터로더 객체 반환."""

        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """평가 데이터를 위한 데이터로더 객체 반환."""

        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        """예측 데이터를 위한 데이터로더 객체 반환."""

        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
