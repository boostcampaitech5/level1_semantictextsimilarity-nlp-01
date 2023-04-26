import argparse
from argparse import Namespace
import json

def parse_arguments() -> argparse.Namespace:
    """config.json 파일의 내용을 argparse.Namespace 객체로 변환.
    
    Returns:
        args (argparse.Namespace): config.json 파일의 내용을 포함하는 Namespace 객체.
    """
    
    with open('./config.json', 'r') as f:
        config = json.load(f)
        
    args = Namespace(**config)
    return args
