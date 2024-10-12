
import h5py as hp
import numpy as np
import pandas as pd
import os
import pdb
import argparse
import random
import sklearn
import scipy
from scipy.signal.windows import hann
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy
import re
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as la
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
def filtering(data_ecog, filter_s = 'delta'):
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
    if filter_s == 'delta_filter':
        return delta_filter
    elif filter_s == 'theta_filter':
        return theta_filter
    elif filter_s == 'alpha_filter':
        return alpha_filter
    elif filter_s == 'beta_filter':
        return beta_filter
    elif filter_s == 'gamma_filter':
        return gamma_filter
    # return delta_filter

def load_dataset(sub_id):
    '''
    returns a list with the paths to h5 files
    '''
    h5files=[]
    sub_path=f'/home/remotelab/sid/codebase/data/extended_dataset/{sub_id}' #provide the path to ECoG dataset

    for h5 in os.listdir(sub_path):
        if h5.endswith('.h5'):
            h5files.append(os.path.join(sub_path, h5))
    return h5files


def sliding_window_augmentation(data, sub_id):
    #Experiment with different window sizes, fs = 500Hz, 30s = 15000samples, total samples = 300*500 = 150000
    if sub_id == '6c29e3':
        window_size = 15360
        stride = 15360
    else:
        window_size = 15000
        stride = 15000
    final_arrays = []
    hann_window = hann(window_size, sym=False)

    for i in range(data.shape[-1]//window_size):
        subarray = data[:, stride * i:stride * (i + 1)]
        subarray_hann = subarray * hann_window
        final_arrays.append(subarray_hann)

    return np.array(final_arrays)

def forward(sub_id):
    pain_data_train=[]
    nopain_data_train = []
    pain_data_test=[]
    nopain_data_test = []
    h5files = load_dataset(sub_id)

    combined_matrix_data = []
    combined_scores = []
    for index in tqdm.tqdm(range(len(h5files))): #/Users/sidharth/Desktop/Herron_lab/spring_2024/data/extended_dataset/0b5a2e/0b5a2e_2.h5
        file_name = h5files[index].split('/')[-1]

        day = file_name.split('.h5')[0].split('_')[-1]
        matpath=h5files[index]
        data_h5=hp.File(matpath,'r')
        matrix = data_h5['neural_windows'][()]
        
        combined_matrix_data.append(matrix)
        # scores = data_h5['intensity'][:]

        # combined_scores.append(scores)
    # breakpoint()
    combined_matrix_data = np.concatenate(combined_matrix_data)
    total_num_trials = combined_matrix_data.shape[0]
    print(f"Total number of trials in subject ID {sub_id} is ", total_num_trials)
    print(combined_matrix_data.shape)
    augmented_array_data = []
    for j in range(total_num_trials):
        ecog_data= combined_matrix_data[j][~np.all(combined_matrix_data[j] == 0, axis = 1)] #getting non zero channels, always (array([  0, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]),)
        if len(ecog_data)>0:
            augmented_data=sliding_window_augmentation(ecog_data, sub_id)
            augmented_array_data.append(augmented_data) 
    
    return augmented_array_data



if __name__ == '__main__':
    #file = load_dataset()
    # breakpoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help = "Enter the subject id")
    args = parser.parse_args()
    sub_id = args.sub
    augmented_array_data = forward(sub_id)
    # breakpoint()
    corr_matrix_list = []
    #avg_corr_matrix = np.zeros((10, 10))
    filter_s = 'delta_filter'
    for trials in augmented_array_data: #each trial have shape [10, electrodes, time]
        #gamma filtering

        trials = filtering(trials, filter_s)
        trials_data = trials.reshape(trials.shape[0], -1)
        fft_transformed = np.fft.fft(trials_data, axis = 1)
        fft_real = np.abs(fft_transformed)
        correlations_fft = np.corrcoef(fft_real)
        corr_matrix_list.append(correlations_fft)
    
    #save the averaged correlation matrix over all trials within a subject
    # breakpoint()
    corr_stack = np.stack(corr_matrix_list)
    mean_corr = np.mean(corr_stack, axis = 0)
    std_corr = np.std(corr_stack, axis = 0)

    plt.figure(figsize = (10,10))
    sns.heatmap(mean_corr, annot=True)
    plt.title("Averaged correlation across trials")
    plt.show()

    plt.savefig(f'/home/remotelab/sid/Experiments/mean_corr_all_trials/{sub_id}_mean_spectral_density_{filter_s}.png')
    plt.close()
    plt.figure(figsize = (10,10))
    sns.heatmap(std_corr, annot=True)
    plt.title("standard deviation correlation across trials")
    plt.show()

    plt.savefig(f'/home/remotelab/sid/Experiments/mean_corr_all_trials/{sub_id}_std_spectral_{filter_s}_density.png')







   