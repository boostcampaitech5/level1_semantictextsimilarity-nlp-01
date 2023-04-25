import time
import argparse
import urllib.parse
from typing import List, Tuple

import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

import pyderman
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def apply_bt_on_dataframe(index: pd.core.series.Series,
                          tgt_cols: List[str] = ['sentence_1'])\
        -> pd.core.series.Series:
    """데이터프레임의 인덱스를 받아 타깃 칼럼에 대하여 역번역 적용.
    
    데이터프레임의 인덱스에서 iterable 형태로 정의된 타깃 칼럼 tgt_cols에 대해
    papago_back_translate 함수 적용.
    
    Args:
        index: 데이터프레임의 인덱스 방향 시리즈 객체.
        tgt_cols: 역번역을 적용할 칼럼(들).
    
    Returns:
        index: 타깃 칼럼에 대하여 역번역이 적용된 데이터프레임 인덱스.
    
    """
    
    for tgt_col in tgt_cols:
        src_text = index[tgt_col]
        _, tgt_text = papago_back_translate(src_text)
        index[tgt_col] = tgt_text
    return index

def papago_back_translate(src_text: str,
                          src_lang: str = 'ko',
                          via_lang: str = 'en',
                          tgt_lang: str = 'ko')\
        -> Tuple[str, str]:
    """주어진 텍스트를 중간 언어(via_lang)를 거쳐 타깃 언어(tgt_lang)로 역번역.
    
    Args:
        src_text: 주어진 텍스트.
        src_lang: 주어진 텍스트의 언어. 기본값: 'ko' (한국어)
        via_lang: 중간 번역 언어. 기본값: 'en' (영어)
        tgt_lang: 타깃 번역 언어. 기본값: 'ko' (한국어)
        
    Returns:
        via_text: 중간 번역 결과 텍스트.
        tgt_text: 타깃 번역 결과 텍스트.
    """
    
    try:
        # ^과 같은 특수문자는 url에 직접 삽입이 불가하여 인코딩 필요
        # urllib.parse.quote 함수로 url 인코딩 수행 가능
        encoded_src_text = urllib.parse.quote(src_text)
        # from source to intermediate
        driver.get(f'https://papago.naver.com/?'\
                   f'sk={src_lang}&tk={via_lang}&st={encoded_src_text}')
        element = WebDriverWait(driver, 2).until(target_present)
        driver.implicitly_wait(3)
        time.sleep(3)
        via_text = element.text
        
        # url 인코딩
        encoded_via_text = urllib.parse.quote(via_text)
        driver.get(f'https://papago.naver.com/?'\
                   f'sk={via_lang}&tk={tgt_lang}&st={encoded_via_text}')
        element = WebDriverWait(driver, 2).until(target_present)
        driver.implicitly_wait(3)
        time.sleep(3)
        tgt_text = element.text
        return via_text, tgt_text
    except TimeoutException:
        return 'Back translation failed', '역번역 실패'

def back_translate(args):    
    df = pd.read_csv(args.input_csv)
    df = df.progress_apply(apply_bt_on_dataframe,
                           axis=1,
                           tgt_cols=args.tgt_cols)
    
    df.to_csv(args.save_dir)


if __name__ == '__main__':
    # 커맨드 라인 arguments 정의
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv',
                        type=str,
                        default='../data/dev2.csv',
                        help='Input CSV file')
    parser.add_argument('--tgt_cols',
                        nargs='+',
                        type=str,
                        default=['sentence_1'],
                        help='Target columns for back translation')
    parser.add_argument('--via_lang',
                        type=str,
                        default='en',
                        help='Intermediate language for back translation '\
                             '(default: en)')
    parser.add_argument('--save_dir',
                        type=str,
                        default='../data/df_back_translated.csv',
                        help='Output path for the back-translated csv file')
    args = parser.parse_args()
    
    # 동적 웹 스크레이핑을 위한 크롬 웹 드라이버 설치
    # pyderman 라이브러리 설치 필요
    driver_path = pyderman.install(browser=pyderman.chrome)
    print(f'Installed geckodriver driver to path: {driver_path}')
    
    # 웹 드라이버 driver 정의
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service)
    driver.implicitly_wait(2)
    time.sleep(2)
    
    target_present = EC.presence_of_element_located((By.XPATH,
                                                     '//*[@id="txtTarget"]'))
    
    back_translate(args)
    