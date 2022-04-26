import time
import logging
import os
import random
import torch
import numpy as np


def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def get_logger(file_path,file_name):

   timestamp=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
   logger = logging.getLogger('SPCC_logger')
   logger.setLevel(logging.DEBUG)
   formatter=logging.Formatter('%(asctime)s : %(message)s')

   file_handler = logging.FileHandler(file_path+file_name+'.log')
   file_handler.setFormatter(formatter)  
   # file_handler.setFormatter(logging.DEBUG)# 只要logger了都记录在文件中
   logger.addHandler(file_handler)

   # stream_handler=logging.StreamHandler()
   # stream_handler.setFormatter(formatter)
   # # stream_handler.setLevel(logging.INFO)# 只有logger.info了才输出到屏幕
   # logger.addHandler(stream_handler)

   return logger
