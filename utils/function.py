import os
import json
import torch
import random
import datetime
import numpy as np
from transformers import set_seed as transformers_seed
from typing import Optional
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_json(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + file_name, 'w') as f:
        json.dump(data, f)


def save_check_point(model, args, tokenizer=None):
    import warnings
    warnings.simplefilter("ignore")    

    now = datetime.datetime.now()
    new_hour = (now.hour + 8) % 24  
    minute = now.minute  
    timestamp = now.strftime(f"%Y%m%d_{new_hour:02d}_{minute:02d}")
    if args.save_dir is None:
        path = './checkpoint/' + model.config.model_type + '/'
    else:
        path = args.save_dir + model.config.model_type + '/'

    path = path + timestamp + '/'
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    model.save_pretrained(path)
    logging.info(f"Model saved to {path}")

