import csv
from itertools import dropwhile,takewhile
from tqdm import tqdm
import pandas as pd
import numpy as np


def read_file(path, features_type, chunk_size, sep_):

    '''Read csv file by parts - chunk_size'''

    for chunk in pd.read_csv(path, chunksize = chunk_size, sep = sep_):     

        yield chunk



def split_tsv(chunk,features_type):
    
    '''Return [splited to np.array tsv file[job_id,feature_i,], rows amount in chunk]'''

    if features_type == '2':

        length = len(chunk)

        features_2_id = [list(chunk['features'])[i].split(',')\
                            [1:] for i in range(len(chunk))]

        features_2_id_int = np.array([list(map(int,features_2_id[i]))\
                                        for i in range(len(chunk))])

        job_ids = np.array(chunk['id_job'])



        return np.hstack([job_ids[:,np.newaxis],features_2_id_int]),length
    
    else :
        None
        
    



def process_train_data(train_data,feature_type):
    
    '''Evaluate mean,var,std param's for all train data'''

    chunk_size = []
    columns_summary = []

    for a, i in enumerate(tqdm(train_data)):

        chunk, length = split_tsv(i,feature_type)
        
        features_sum = chunk[:,1:].sum(axis = 0)

        columns_summary.append(features_sum)
        chunk_size.append(length)

        if a == 0 :

            chunk_lake = chunk[:, 1:]

        else :

            chunk_lake = np.append(chunk_lake, chunk[:,1:], axis = 0)
            
        print('\n Processed train-data {} rows'.format(length))

    mean = sum(columns_summary) / sum(chunk_size)
    var = np.array([(chunk_lake[i] - mean)**2  for i in range(sum(chunk_size))], dtype = object)
    std = (sum(var) / sum(chunk_size))**0.5

    return mean, std




def feature_2_stand(test_data,mean,std):
    
    '''Z-score for chunk data'''

    return np.array((test_data - mean) / std)





def process_test_data(test_data, mean, std, sep, features_type):
    
     '''Return writing tsv file with [job_id,feature_i_z_score,max_feature_2_index,max_feature_2_abs_mean_diff]'''

    columns = ['feature_2_stand_{}'.format(str(i)) for i in range(256)]
    columns = ['job_id'] + columns + ['max_feature_2_index','max_feature_2_abs_mean_diff']

    for  i in (tqdm(test_data)):
        

        test_data_chunk, length = split_tsv(i,features_type)

        test_data_stand = feature_2_stand(test_data_chunk[:, 1:], mean, std)

        max_feature_2_index = np.asarray([np.argmin(test_data_chunk[:, 1:][a]) \
            for a in range(len(test_data_chunk))])

        max_feature_2_abs_mean_diff = np.asarray([np.abs(test_data_chunk[:, 1:][m][n] - mean[n]) \
            for m, n in enumerate(max_feature_2_index)])

        _values =  np.hstack([test_data_chunk[:,:1].reshape(len(test_data_chunk) ,1),\
        test_data_stand, max_feature_2_index.reshape(len(max_feature_2_index), 1),\
        max_feature_2_abs_mean_diff.reshape(len(max_feature_2_index), 1)])

        df = pd.DataFrame(_values, columns = columns)

        with open("./test_proc.tsv","a") as f:
            df.to_csv(f, header=f.tell()==0, sep=sep, index=False)

        print('\n Processed test-data {} rows'.format(length))


            
