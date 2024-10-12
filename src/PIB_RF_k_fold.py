import numpy as np
import scipy.linalg as la
import torch
import pdb
import random
import tqdm
import json
import argparse
from utils import parse_json
import scipy.signal as signal
from utils import load_dataset, sliding_window_augmentation, forward, PIB, rf_classification, filtering
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

'''
This script contains the leave one trial out cross validation using PIB_CSP_RF method for all the subjects

The usage of this script is 

python PIB_CSP_RF_k_fold.py --sub 822e28 > /home/remotelab/sid/codebase/code/PainBiomarker/runs/Experiments/PIB_CSP_RF/sub.txt
'''


if __name__ == '__main__':

    #load components from JSON
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help = "Enter the subject id")
    parser.add_argument("--config", help = "location of config file")
    args = parser.parse_args()
    sub_id = args.sub
    config_file = args.config
    cfg = parse_json(config_file)
    
    total_data, total_labels = forward(sub_id) #get sliced data 
    # breakpoint()
    grouped_data = total_data.reshape(total_data.shape[0]//30, 30, total_data.shape[1], 6)
    grouped_labels = total_labels.reshape(total_labels.shape[0]//30, -1) #Ensures one of each trial goes to either train or test, no data leakage
    kf = KFold(n_splits = grouped_data.shape[0]) # leave one trial out cross validation
    kf.get_n_splits(grouped_data)
    mean_accuracies_fold = []
    print(f"Total number of folds = {grouped_data.shape[0]}")

    components_list = cfg.csp.components

    if components_list == ['max']:
        componenets_list = [grouped_data.shape[-2]]
    if components_list == ['full']:
        components_list = [2**i for i in range(1,int(grouped_data.shape[-2]).bit_length())]
        components_list.append(grouped_data.shape[-2])
    # components_list = ['full']
    for components in components_list:

        for i, (train_indices, test_indices) in enumerate(kf.split(grouped_data)):
            #print(f"Fold {i}")
            # breakpoint()
            train_arr = grouped_data[train_indices]
            train_arr = train_arr.reshape(-1, train_arr.shape[-2], train_arr.shape[-1])
            y_train = grouped_labels[train_indices]
            y_train = y_train.reshape(-1)


            pos_indices = [ind for ind in range(len(y_train)) if y_train[ind] == 1]
            neg_indices = [ind for ind in range(len(y_train)) if y_train[ind] == 0]

            # breakpoint()
            pain_data_tr = train_arr[pos_indices]
            nopain_data_tr = train_arr[neg_indices]

            # print(f"pain data shape train : {pain_data_tr.shape}")
            # print(f"No pain data shape train: {nopain_data_tr.shape}")
        
            test_arr = grouped_data[test_indices]
            test_arr = test_arr.reshape(-1, test_arr.shape[-2], test_arr.shape[-1])
            y_test = grouped_labels[test_indices]
            y_test = y_test.reshape(-1)

            train_set = train_arr
            test_set = test_arr

            train_set = train_set.reshape(train_set.shape[0],-1) 
        
            test_set = test_set.reshape(test_set.shape[0],-1) 

            max_mean_acc = rf_classification(train_set.real, test_set.real, y_train, y_test, sub_id)
            mean_accuracies_fold.append(max_mean_acc)

            # print(f"Maximum mean accuracy for subject {sub_id} for Fold {i} is {max_mean_acc}")
        # components = grouped_data.shape[-2]
        print(f"Mean accuracy for subject CSP {sub_id} is {np.mean(mean_accuracies_fold)} with components of CSP = {components}")

    

