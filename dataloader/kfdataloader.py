import pandas as pd
import pytorch_lightning as pl

from .dataset import *

from tqdm.auto import tqdm
from datasets import load_dataset
from typing import List, Tuple

from transformers import AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
pl.seed_everything(420)


class KfoldDataloader(pl.LightningDataModule):
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
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        self.columns = ['id', 'source', 'sentence_1', 'sentence_2', 
                        'label', 'binary-label']
        
        self.total_data, self.test_data = self.redistribute_train_and_test()


    def tokenizing(self, dataframe: pd.DataFrame) -> List[str]:
        """Tokenize concatenated sentences.
        
        Concatenate two pairs of sentences with [SEP] token in between
        and tokenize the resulting string, then append it to a list.

        Args:
            df: A dataframe with columns: 'source', 'sentence1', 'sentence2', 
                'label', and 'binary_label'.
                
        Returns:
            data: A list of tokens obtained by tokenizing the two sentences 
                concatenated by the [SEP] token.
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

    def preprocessing(self, data: pd.DataFrame) \
            -> Tuple[List[str], List[float or int]]:
        """Preprocess a dataframe.

        Extract sentences along with their labels from the given dataframe
        and stor them in separate lists for preprocesing.

        Args:
            data: A dataframe.

        Returns:
            inputs: A list of tokens obtained by tokenizing the two sentences 
                concatenated by the [SEP] token.
            targets: A list containing target labels.
        """
        
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

    def setup(self, stage='fit'):
        # 데이터셋 로드
        # revision: tag name, or branch name, or commit hash
        total_data, test_data = self.redistribute_train_and_test()
        
        if stage == 'fit':
            # 데이터 준비
            total_input, total_targets = self.preprocessing(total_data)
            total_dataset = Dataset(total_input, total_targets)

            # 데이터셋 num_splits번 fold
            kf = KFold(n_splits=self.num_folds, 
                       shuffle=self.shuffle)
            all_splits = [k for k in kf.split(total_dataset)]
            
            # k번째 fold 된 데이터셋의 index 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()

            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes] 
            self.val_dataset = [total_dataset[x] for x in val_indexes]

        else:
            # 평가데이터 준비
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)
            self.predict_dataset = Dataset(test_inputs, [])

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
        
    def redistribute_train_and_test(self):
        train = load_dataset("Salmons/STS_Competition", 
                            split='train', 
                            column_names=self.columns, 
                            revision=self.dataset_commit_hash)
        valid = load_dataset("Salmons/STS_Competition", 
                            split='validation', 
                            column_names=self.columns, 
                            revision=self.dataset_commit_hash)
        
        # 'id' omitted.
        # ['source', 'sentence_1', 'sentence_2', 'label', 'binary-label']
        train_data = train.to_pandas().iloc[1:].reset_index(drop=True)\
                          .astype({'label': 'float', 'binary-label': 'float'})
        val_data = valid.to_pandas().iloc[1:].reset_index(drop=True)\
                        .astype({'label': 'float', 'binary-label': 'float'})
        
        combined_data = pd.concat([train_data, val_data], axis=0)
        
        total_data, test_data, _, _ = train_test_split(
          combined_data,
          combined_data[self.target_columns],
          train_size=self.train_ratio,
          stratify=combined_data[self.target_columns],
        )
        return total_data, test_data