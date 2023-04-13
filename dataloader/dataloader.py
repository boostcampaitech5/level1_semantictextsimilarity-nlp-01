# dataloader.py
import pandas as pd

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer

from .dataset import *
from tqdm.auto import tqdm

# huggingface datasets
from datasets import load_dataset

class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, dataset_commit_hash):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_commit_hash = dataset_commit_hash

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       max_length=160)
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), 
                              desc='tokenizing', 
                              total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text] for text in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, 
                                     padding='max_length', truncation=True)
            data.append(outputs['input_ids'])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)
        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 데이터셋 로드
            # revision: tag name, or branch name, or commit hash
            train = load_dataset("Salmons/STS_Competition", 
                                split='train', 
                                column_names=['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label'], 
                                revision=self.dataset_commit_hash)
            valid = load_dataset("Salmons/STS_Competition", 
                                split='validation', 
                                column_names=['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label'], 
                                revision=self.dataset_commit_hash)

            # pandas 형식으로 변환
            train_data = train.to_pandas().iloc[1:].reset_index(drop=True).astype({'label':'float', 'binary-label':'float'})
            val_data = valid.to_pandas().iloc[1:].reset_index(drop=True).astype({'label':'float', 'binary-label':'float'})

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        
        else:
            test = load_dataset("Salmons/STS_Competition", 
                                split='validation', 
                                column_names=['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label'], 
                                revision=self.dataset_commit_hash)
            predict = load_dataset("Salmons/STS_Competition", 
                                split='test', 
                                column_names=['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label'], 
                                revision=self.dataset_commit_hash)

            test_data = test.to_pandas().iloc[1:].reset_index(drop=True).astype({'label':'float', 'binary-label':'float'})
            predict_data = predict.to_pandas().iloc[1:].reset_index(drop=True)[['id', 'source', 'sentence_1', 'sentence_2']]

            test_inputs, test_targets = self.preprocessing(test_data)
            predict_inputs, predict_targets = self.preprocessing(predict_data)

            self.test_dataset = Dataset(test_inputs, test_targets)            
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, 
                                           batch_size=self.batch_size, 
                                           shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, 
                                           batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, 
                                           batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, 
                                           batch_size=self.batch_size)