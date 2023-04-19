# dataloader.py
import pandas as pd
import re
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from .dataset import *
from tqdm.auto import tqdm
from datasets import load_dataset
pl.seed_everything(420)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle,
                 dataset_commit_hash, use_val_for_predict=False):
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
        self.columns = ['id', 'source', 'sentence_1', 'sentence_2', 
                        'label', 'binary-label']

        self.use_val_for_predict = use_val_for_predict
    
    def preprocess_text(self, text):
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
        # 한글, 영어, 숫자, 특수문자 ? ; : ^만 남기기
        text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.,?!:;^]', '', text)
        # ㅎ, ㅋ, ㅠ 제거
        text = re.sub('[ㅋㅎㅠ]+', '', text)
        # 특수문자 중복제거 
        text = re.sub(r'([.?!,:]){2,}', r'\1', text)
        # 양 공백 제거 
        text = text.strip()
        # 반복되는 글자를 2개로 줄여준다. 
        text = re.sub(r"(.)\1+", r"\1\1", text)

        return text

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

    def preprocessing(self, data, is_train = False):
        # reverse augmentation
        if is_train == True:
            df_2 = data.copy()
            df_2['sentence_1'] = data['sentence_2']
            df_2['sentence_2'] = data['sentence_1']
            data = pd.concat([data, df_2], ignore_index=True)
            data = data.sample(frac=1, random_state=420).reset_index(drop=True)
            del df_2

        # 기본 텍스트 전처리
        data[self.text_columns] = data[self.text_columns].apply(
          lambda x: x.apply(self.preprocess_text))

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
                                column_names=self.columns, 
                                revision=self.dataset_commit_hash)
            valid = load_dataset("Salmons/STS_Competition", 
                                split='validation', 
                                column_names=self.columns, 
                                revision=self.dataset_commit_hash)

            # pandas 형식으로 변환
            train_data = train.to_pandas().iloc[1:].reset_index(drop=True)\
                              .astype({'label':'float', 'binary-label':'float'})
            val_data = valid.to_pandas().iloc[1:].reset_index(drop=True)\
                            .astype({'label':'float', 'binary-label':'float'})

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다. 
            # 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
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
                            .astype({'label':'float', 'binary-label':'float'})
            
            # val_dataset으로 predict 해서 결과물 보려고 구분하는 지점
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