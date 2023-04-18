#!pip install googletrans==4.0.0-rc1

import time
import pandas as pd
from tqdm import tqdm
from googletrans import Translator as google_trans

# 데이터 불러오기
train_data = pd.read_csv("/opt/ml/data/train.csv")
dev_data = pd.read_csv("/opt/ml/data/dev.csv")
test_data = pd.read_csv("/opt/ml/data/test.csv")

# 번역 함수 정의
def translate(translator, text_org, src_lang='ko', dest_lang='en'):
    text_tgt = ''
    try:
        text_tgt = translator.translate(text_org, src=src_lang, dest=dest_lang).text
    except AttributeError: pass
    except TypeError: pass
    except JSONDecodeError: pass
    except NameError: pass
    return text_tgt

def translate_column(column, device):
    translated = []
    for idx, s in enumerate(tqdm(column)):
        translated.append(translate(device, s))
        if idx % 50 == 0:
            time.sleep(2)
    return translated

device = google_trans()


dev_data['sentence_1_eng'] = translate_column(dev_data['sentence_1'], device)
dev_data['sentence_2_eng'] = translate_column(dev_data['sentence_2'], device)
dev_data.to_csv('/opt/ml/data/dev_translated.csv')

test_data['sentence_1_eng'] = translate_column(test_data['sentence_1'], device)
test_data['sentence_2_eng'] = translate_column(test_data['sentence_2'], device)
test_data.to_csv('/opt/ml/data/test_translated.csv')

train_data['sentence_1_eng'] = translate_column(train_data['sentence_1'], device)
train_data['sentence_2_eng'] = translate_column(train_data['sentence_2'], device)
train_data.to_csv('/opt/ml/data/train_translated.csv')