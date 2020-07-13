
import csv
from itertools import dropwhile,takewhile
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import read_file,process_train_data,process_test_data
import argparse

parser = argparse.ArgumentParser(description = 'Feature engineering')

parser.add_argument('--train_path',type = str)
parser.add_argument('--test_path',type = str)

# parser.add_argument('--feature_type',type = str)
parser.add_argument('--chunk_size',type = int,default= 500)
#parser.add_argument('--sep',type = str)
args = parser.parse_args()



if True :
   
    TRAIN_PATH = args.train_path 
    TEST_PATH =  args.test_path
    FEATURES_TYPE = "2"
    CHUNK_SIZE = int(args.chunk_size) 
    SEP = "\t"

    test_data = read_file(TEST_PATH,FEATURES_TYPE,CHUNK_SIZE, SEP)
    train_data = read_file(TRAIN_PATH, FEATURES_TYPE, CHUNK_SIZE, SEP)

    MEAN,STD = process_train_data(train_data,FEATURES_TYPE)
    process_test_data(test_data,MEAN, STD,SEP,FEATURES_TYPE)
    
  
    

    