
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
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.linalg as la
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import json
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint



def parse_json(config_file):
    with open(config_file, 'r') as files:
        cfg = json.load(files)
        return cfg


def CSP(*tasks):
        
      if len(tasks) < 2:
        print("Must have at least 2 tasks for filtering.")
        return (None,) * len(tasks)
      else:
        filters = ()
        # CSP algorithm
        # For each task x, find the mean variances Rx and not_Rx, which will be used to compute spatial filter SFx
        iterator = range(0,len(tasks))
        for x in iterator:
                          
          # Find Rx
          # breakpoint()
          Rx = covarianceMatrix(tasks[x][0])

          for t in range(1,len(tasks[x])):
            Rx += covarianceMatrix(tasks[x][t])
          Rx = Rx / len(tasks[x])

          # Find not_Rx
          count = 0
          not_Rx = Rx * 0
          for not_x in [element for element in iterator if element != x]:
            for t in range(0,len(tasks[not_x])):
              not_Rx += covarianceMatrix(tasks[not_x][t])
              count += 1
          not_Rx = not_Rx / count

          # Find the spatial filter SFx
        #   breakpoint()
          SFx = spatialFilter(Rx,not_Rx)
          filters += (SFx,)

          # Special case: only two tasks, no need to compute any more mean variances
          if len(tasks) == 2:
            filters += (spatialFilter(not_Rx,Rx),) #getting nopain filter : one is the opposite of the other

            break
        return filters

# covarianceMatrix takes a matrix A and returns the covariance matrix, scaled by the variance
def covarianceMatrix(A):
	Ca = np.dot(A,np.transpose(A))/np.trace(np.dot(A,np.transpose(A)))
	return Ca

# spatialFilter returns the spatial filter SFa for mean covariance matrices Ra and Rb
def spatialFilter(Ra,Rb):
	R = Ra + Rb
	E,U = la.eig(R)
	# CSP requires the eigenvalues E and eigenvector U be sorted in descending order
	ord = np.argsort(E)
	ord = ord[::-1] # argsort gives ascending order, flip to get descending
	E = E[ord]
	U = U[:,ord]

	# Find the whitening transformation matrix
	P = np.dot(np.sqrt(la.inv(np.diag(E))),np.transpose(U))

	# The mean covariance matrices may now be transformed
	Sa = np.dot(P,np.dot(Ra,np.transpose(P)))
	Sb = np.dot(P,np.dot(Rb,np.transpose(P)))
    


	# Find and sort the generalized eigenvalues and eigenvector
	E1,U1 = la.eig(Sa,Sb)
	ord1 = np.argsort(E1)
	ord1 = ord1[::-1]
	E1 = E1[ord1]
	U1 = U1[:,ord1]

	# The projection matrix (the spatial filter) may now be obtained
	SFa = np.dot(np.transpose(U1),P)
	return SFa.astype(np.float32)

def pain_binary(scores, threshold):  
    binarizied_scores = []

    #------strict approach---------------------
    for score in scores:
        if score>threshold:
            binarizied_scores.append(1)
        elif score<=threshold:
            binarizied_scores.append(0)

 
    return np.array(binarizied_scores)



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

def PIB(data):
    # Apply filtering to data
    filter_series = filtering(data)  # List of 6 filtered bands, each shape (B, C, T)
    
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



def sliding_window_augmentation(data, sub_id):
    #Experiment with different window sizes, fs = 500Hz
    if sub_id != '6c29e3':
        window_size = 5000
        stride = 5000
    else:
        window_size = 5120
        stride = 5120

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
        if sub_id != '6c29e3':
            day = file_name.split('.h5')[0].split('_')[-1]
        else:
            dayX=file_name.split('_')[1]
            day_match = re.search(r'\d+', dayX)
            day = day_match.group() if day_match else None 
        matpath=h5files[index]
        data_h5=hp.File(matpath,'r')
        matrix = data_h5['neural_windows'][()]
        
        combined_matrix_data.append(matrix)
        if sub_id != '6c29e3':
            scores = data_h5['intensity'][:]
            combined_scores.append(scores)
        if sub_id == '6c29e3':
            # breakpoint()
            csv_scores = survey_response = pd.read_excel("/home/remotelab/sid/codebase/data/VAS_timestamps/6c29e3_MPQ_survey_responses.xlsx", sheet_name = None)
            total_scores = csv_scores['Sheet1'][csv_scores['Sheet1']['day'] == int(day)]['Total'].tolist()
            combined_scores.append(total_scores)
        
    combined_scores = np.concatenate(combined_scores)
    # breakpoint()
    #---------------c5a5e9-----------------------------------------------------

    if sub_id=='c5a5e9':
        '''
        #x<=5 low pain : 35/66 
        OrderedDict({np.float64(0.0): 1, np.float64(2.0): 6, np.float64(3.0): 6, np.float64(4.0): 10, \
        np.float64(5.0): 12, np.float64(6.0): 11, np.float64(7.0): 10, np.float64(8.0): 3, np.float64(9.0): 3, np.float64(10.0): 4})

        The scores are kind of well distributed majorly centered around mid range [4, 7] with some values in the extremes.
        Hence in this case it is clear that 5 becomes a good threshold. hence I am choosing <=5 to be low pain (35/66)
        '''
        threshold = 5
        ignore_scores = [4,5,6,7]
        
        ignore_indices = [x for x in range(len (combined_scores)) if combined_scores[x] in ignore_scores]
        selected_indices = [x for x in range(len(combined_scores)) if x not in ignore_indices]
        # breakpoint()


#--------------------------------------------------------------------------------
#------------------822e28-------------------------------------------------------

    if sub_id=='822e28':
        '''
        #0 is a good threshold, 19/36 scores are 0 and clearly indicates nopain

        OrderedDict({np.float64(0.0): 19, np.float64(1.0): 1, np.float64(2.0): 1, np.float64(3.0): 6, np.float64(5.0): 5, np.float64(6.0): 4})
        Patient majorly felt no pain as per the scores, which means the other values except for 0 clearly indicates pain hence 0 is a good threshold
        >0 is pain 
        But poor representation of positive labels
        '''
        threshold = 1
        ignore_scores = [1,2,3]
        ignore_indices = [x for x in range(len (combined_scores)) if combined_scores[x] in ignore_scores]
        selected_indices = [x for x in range(len(combined_scores)) if x not in ignore_indices]


#-------------------------------------------------------------------------------
#------------------422bc5-------------------------------------------------------

    if sub_id == '422bc5':
        '''
        #the data is distributed from [2,8], OrderedDict({np.float64(2.0): 1, np.float64(3.0): 3, np.float64(4.0): 14, np.float64(5.0): 14, np.float64(7.0): 6, np.float64(8.0): 17})
        Patient has severly moderate high pain (8) as per the scores, also apart from this, the patient has felt more in the normal region (not pain, not no-pain) hence <=5 is nopain as per my suggestion
        # <=5 is low pain
        '''
        threshold = 5
        ignore_scores = [4,5,6,7]
        ignore_indices = [x for x in range(len (combined_scores)) if combined_scores[x] in ignore_scores]
        selected_indices = [x for x in range(len(combined_scores)) if x not in ignore_indices]



#-------------------------------------------------------------------------------

#------------------0b5a2e-------------------------------------------------------

    if sub_id=='0b5a2e':
        '''
        OrderedDict({np.float64(2.0): 5, np.float64(3.0): 11, np.float64(4.0): 22, np.float64(5.0): 16, np.float64(6.0): 5, np.float64(7.0): 7, np.float64(8.0): 4})
        very less times the patient has felt extreme pains, most of the times, the patient has felt moderate pain [4 and 5]. 
        The model will obviously have hard time classifying it into high or low pain states, but might slightly do better in classifying low pain better
        because of the distribution (3 has good representation)
        Hence I am selecting the threshold to be >=5 as high pain
        
        '''
        threshold = 4
        ignore_scores = [4,5,6, 7]
        ignore_indices = [x for x in range(len (combined_scores)) if combined_scores[x] in ignore_scores]
        selected_indices = [x for x in range(len(combined_scores)) if x not in ignore_indices]

    
    if sub_id == '6c29e3':
        '''
        The sorted scores are array([ 41,  43,  43,  44,  46,  48,  48,  50,  53,  56,  60,  60,  62, 89,  98, 117]). It is distributed in the range [41,117]
        The median turns out to be 51.5. There is atleast 12 score difference between right half and left half if we ignore 50,53,56
        The unsorted are array([ 46,  44,  48,  98,  60,  56,  43,  43,  41,  62, 117,  89,  60, 48,  50,  53]) 

        '''
        threshold = 80
        ignore_scores = [48, 50,53,56,60,62]
        ignore_indices = [x for x in range(len (combined_scores)) if combined_scores[x] in ignore_scores]
        selected_indices = [x for x in range(len(combined_scores)) if x not in ignore_indices]
         
     
#-------------------------------------------------------------------------------


    # combined_scores = np.concatenate(combined_scores)
    combined_matrix_data = np.concatenate(combined_matrix_data, axis = 0)
    combined_scores_selected = combined_scores[selected_indices]
    combined_matrix_data_selected = combined_matrix_data[selected_indices]
    labels = pain_binary(combined_scores_selected, threshold)
    TOTAL_DATA = []
    scores = []
    # breakpoint()
    for j in range(combined_matrix_data_selected.shape[0]):
        ecog_data = combined_matrix_data_selected[j][~np.all(combined_matrix_data_selected[j]==0, axis = 1)]
        augmented_data=sliding_window_augmentation(ecog_data, sub_id)
        for subarrays in augmented_data:
            data = subarrays
            data = PIB(subarrays)
            if data.shape[0] == 0:
                continue
            TOTAL_DATA.append(data)
            scores.append(labels[j])
    # breakpoint()
    TOTAL_DATA = np.array(TOTAL_DATA)
    scores = np.array(scores)
    # breakpoint()
    return TOTAL_DATA, scores

#The data needs to be balanced to apply csp filters

def calc_csp(pain_data_train, nopain_data_train, test_data, components = 'full'):
    '''
    pain_data_train #[B, C, 6] #(900,90,6)
    nopain_data_train #[B, C, 6]
    pain_data_test #[B, C, 6]
    nopain_data_test #[B, C, 6]

    Function description : Calculates the spatial filter based on the pain and nopain data from the train set and 
                           applying it to both train set and the test set
    '''

    csp_filter = CSP(pain_data_train, nopain_data_train) #csp_filter[0] is pain, csp_filter[1] is nopain
    #csp_filter has shape (2, #vectors, components) 
    if components != 'full':
        csp_filter[0] = np.concatenate([csp_filter[0][:, :components//2], csp_filter[0][:, -components//2:]], axis = 0)
        csp_filter[1] = np.concatenate([csp_filter[1][:, :components//2], csp_filter[1][:, -components//2:]], axis = 0)
    


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

    test_filtered = test_filtered_pos + test_filtered_neg



    test_set = test_filtered
    train_set = np.concatenate([train_pain_filtered, train_nopain_filtered])

    return train_set, test_set
     


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