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
from utils import load_dataset, sliding_window_augmentation, forward, PIB
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



def PIB(data):
    # Apply filtering to data
    filter_series = data  # List of 6 filtered bands, each shape (B, C, T)
    
    # Initialize empty list to store power features
    feature_power = []

    # Calculate total power in each band for each trial and each electrode
    for i in range(len(filter_series)):  # Iterate over each filtered band
        power_band = np.sum(np.abs(signal.hilbert(filter_series[i], axis=-1))**2, axis=-1)
        feature_power.append(power_band)  # Collect power for current band
    
    # Stack features for all bands, shape will be (B, C, 6) after stacking
    feature_power = np.stack(feature_power, axis=-1)  # Concatenates along new axis for frequency bands
    
    return feature_power

def filtering(data_ecog):
    # Butterworth filter for each band
    sos_delta = signal.butter(2, [0.1, 4], btype='bandpass', analog=False, output='sos', fs=500)
    sos_theta = signal.butter(2, [4, 8], btype='bandpass', analog=False, output='sos', fs=500)
    sos_alpha = signal.butter(2, [8, 13], btype='bandpass', analog=False, output='sos', fs=500)
    sos_beta = signal.butter(2, [13, 30], btype='bandpass', analog=False, output='sos', fs=500)
    sos_gamma = signal.butter(2, [30, 60], btype='bandpass', analog=False, output='sos', fs=500)
    sos_highgamma = signal.butter(2, [60, 200], btype='bandpass', analog=False, output='sos', fs=500)
    
    # Apply filtering to each frequency band for all trials/channels
    delta_filter = signal.sosfilt(sos_delta, data_ecog, axis=-1)
    theta_filter = signal.sosfilt(sos_theta, data_ecog, axis=-1)
    alpha_filter = signal.sosfilt(sos_alpha, data_ecog, axis=-1)
    beta_filter = signal.sosfilt(sos_beta, data_ecog, axis=-1)
    gamma_filter = signal.sosfilt(sos_gamma, data_ecog, axis=-1)
    highgamma_filter = signal.sosfilt(sos_highgamma, data_ecog, axis=-1)
    
    # Return list of filtered signals for each band
    return [delta_filter, theta_filter, alpha_filter, beta_filter, gamma_filter, highgamma_filter]

def spatial_filter(Ra, Rb):
    R = Ra + Rb
    E, U = la.eig(R)
    order = np.argsort(E)[::-1]

    E, U = E[order], U[:, order]
    # P = E^{-1/2} U.T
    P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

    # The mean covariance matrices may now be transformed
    Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
    Sb = np.dot(P,np.dot(Rb,np.transpose(P)))

    #Solve the generalized eigenvalue problem to separate the data 
    E1, U1 = la.eig(Sa, Sb)
    order1 = np.argsort(E1)[::-1]
    E1, U1 = E1[order1], U1[:, order1]

    #Compute the filter
    SFilter = np.dot(np.transpose(U1),P) #rows are the components (components, num_vecs)
    return (P,SFilter.astype(np.float32))

def covarianceMatrix(A):
    return np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))

def CSP(*tasks):
    '''
    CSP for 2 classes only
    Args:
        tasks: list of data for each class required to be separated
        task should be of the format (B, C, feats)
    Returns
        spatial_filter
    '''
    assert len(tasks) == 2, "Require 2 classes"
    
    pos, neg = tasks[0], tasks[1]
    # breakpoint()
    unnormalized_rx = np.array([covarianceMatrix(x) for x in pos])
    unnormalized_not_rx = np.array([covarianceMatrix(x) for x in neg])
    rx = np.mean(unnormalized_rx, axis = 0)
    not_rx = np.mean(unnormalized_not_rx,axis = 0)
    # breakpoint()
    whitening_filter, pos_filter = spatial_filter(rx, not_rx)
    whitening_filter, neg_filter = spatial_filter(not_rx, rx)

    
    #Return the filters pos and neg
    return (pos_filter, neg_filter)

def calc_csp(pain_data_train, nopain_data_train, test_data, components = 'full'):
    '''
    pain_data_train #[B, C, 6] #(900,90,6)
    nopain_data_train #[B, C, 6]
    pain_data_test #[B, C, 6]
    nopain_data_test #[B, C, 6]

    Function description : Calculates the spatial filter based on the pain and nopain data from the train set and 
                           applying it to both train set and the test set[]
    '''

    csp_filter = CSP(pain_data_train, nopain_data_train) #csp_filter[0] is pain, csp_filter[1] is nopain
    #csp_filter has shape (2, components, num_vecs) 
    if components != 'max':
        
        csp_filter_mod = []
        csp_filter_mod.append(np.concatenate([csp_filter[0][:components//2, :], csp_filter[0][-components//2:, :]], axis = 0))
        csp_filter_mod.append(np.concatenate([csp_filter[1][:components//2, :], csp_filter[1][-components//2: :]], axis = 0))
        csp_filter = np.asarray(csp_filter_mod)
    train_pain_filtered = np.einsum('ij,bjk->bik', csp_filter[0], pain_data_train)
    train_pain_filtered /= np.linalg.norm(train_pain_filtered, axis = (1,2), keepdims=True)

    train_nopain_filtered = np.einsum('ij,bjk->bik', csp_filter[1], nopain_data_train)
    train_nopain_filtered /= np.linalg.norm(train_nopain_filtered, axis = (1,2), keepdims=True)

    #--------------test_data_filtering________________________
    '''
    Applying both filters and taking average
    '''

    test_filtered_pos = np.einsum('ij,bjk->bik', csp_filter[0], test_data)
    test_filtered_pos /= np.linalg.norm(test_filtered_pos, axis = (1,2), keepdims=True) 

    test_filtered_neg = np.einsum('ij,bjk->bik', csp_filter[1], test_data)
    test_filtered_neg /= np.linalg.norm(test_filtered_neg, axis = (1,2), keepdims=True) 


    test_filtered = np.mean([test_filtered_pos, test_filtered_neg], axis = 0)



    test_set = test_filtered
    train_set = np.concatenate([train_pain_filtered, train_nopain_filtered])
    y_train = np.concatenate([np.ones(len(train_pain_filtered)), np.zeros(len(train_nopain_filtered))])

    return train_set, test_set, y_train


def rf_classification(train_set, test_set, y_train, y_test, sub_id):
    classifier = RandomForestClassifier(random_state = 42)
    classifier.fit(train_set, y_train)
    num_iterations = 100
    acc = []
    prec = []
    rec = []
    std_acc=[]
    std_prec=[]
    std_rec=[]
    acc_max=[]
    acc_min=[]
    Y_PRED=[]
    ACC=[]
    test_sample_sizes = np.arange(1, len(test_set), 1)

    for sample_size_index in range(len(test_sample_sizes)):
        accuracy_sample_size = []
        for iter in range(num_iterations):
            indices = random.choices(range(len(test_set)), k=test_sample_sizes[sample_size_index])
            test_data_boot = test_set[indices]
            y_test_boot = y_test[indices]
            y_pred = classifier.predict(test_data_boot)
            Y_PRED.append(y_pred)
            accuracy = accuracy_score(y_test_boot, y_pred)
            accuracy_sample_size.append(accuracy)
        ACC.append(accuracy_sample_size)
        acc.append(np.mean(accuracy_sample_size))
        std_acc.append(np.std(accuracy_sample_size))
        acc_max.append(max(accuracy_sample_size))
        acc_min.append(min(accuracy_sample_size))
    #print(f"Maximum mean accuracy of {sub_id} is {np.max(acc)}")
    return np.max(acc)

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
            # print(f"test data shape : {test_arr.shape}")
            # print(f" pos labels in test : {len([x for x in range(len(y_test)) if y_test[x] == 1])}")
            # print(f" neg labels in test : {len([x for x in range(len(y_test)) if y_test[x] == 0])}")
            # breakpoint()
            train_set, test_set, y_train = calc_csp(pain_data_tr, nopain_data_tr, test_arr, components)

            # train_set = train_arr
            # test_set = test_arr

            train_set = train_set.reshape(train_set.shape[0],-1) 
        
            test_set = test_set.reshape(test_set.shape[0],-1) 

            max_mean_acc = rf_classification(train_set.real, test_set.real, y_train, y_test, sub_id)
            mean_accuracies_fold.append(max_mean_acc)

            # print(f"Maximum mean accuracy for subject {sub_id} for Fold {i} is {max_mean_acc}")
        # components = grouped_data.shape[-2]
        print(f"Mean accuracy for subject CSP {sub_id} is {np.mean(mean_accuracies_fold)} with components of CSP = {components}")

    

