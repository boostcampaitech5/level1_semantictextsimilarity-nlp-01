from argparse import Namespace
import json

def parse_arguments():
    
    with open('./config.json', 'r') as f:
        config = json.load(f)
        
    args = Namespace(**config)
    return args
