
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
from torchvision import transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint


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

def pain_binary(scores):  
    #rejecting scores [4,5,6] 
    
    #matlab_fle = load_dataset(path) #<KeysViewHDF5 ['channels', 'file', 'fs', 'intensities', 'neural_windows', 'target_sample', 'timestamps']>
    #reject_scores = [4,5,6]
    # high_threshold = 6
    # low_threshold = 4
    # threshold = 5
    binarizied_scores = []
    #------median approach----------------------
    for score in scores:
        if score>np.median(scores):
            binarizied_scores.append(1)
        elif score<np.median(scores):
            binarizied_scores.append(0)
    #------strict approach---------------------
    # for score in scores:
    #     if score>threshold:
    #         binarizied_scores.append(1)
    #     elif score<threshold:
    #         binarizied_scores.append(0)



    # breakpoint()
    #--------------------balanced_scores_approach------------------
    pos_count = len([x for x in scores if x > np.median(scores)])
    neg_count = len([x for x in scores if x < np.median(scores)])
    for score in scores:
        
        if score == np.median(scores):

            # breakpoint()
            if pos_count<neg_count:
                binarizied_scores.append(1)
                pos_count+=1
            else:
                binarizied_scores.append(0)    
                neg_count+=1
    #----------------Unbalanced_scores_apprach--------------------
    # pos_count = len([x for x in scores if x > threshold])
    # neg_count = len([x for x in scores if x < threshold])
    # for score in scores:
        
    #     if score == threshold:

    #         # breakpoint()
    #         if pos_count<neg_count:
    #             binarizied_scores.append(1)
    #             pos_count+=1
    #         else:
    #             binarizied_scores.append(0)    
    #             neg_count+=1
    # breakpoint()          
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



def sliding_window_augmentation(data):
    #Experiment with different window sizes, fs = 500Hz
    window_size = 5000
    stride = 5000
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
        scores = data_h5['intensity'][:]
        combined_scores.append(scores)
        
        # breakpoint()

    # print(combined_scores)
    breakpoint()
    
    #---------------c5a5e9-----------------------------------------------------
    #c5a5e9 : get [0][0] and [0][1] as high pain and  [3][5:7] as low pain
    if sub_id=='c5a5e9':
        '''
        #x<=5 low pain : 35/66 
        OrderedDict({np.float64(0.0): 1, np.float64(2.0): 6, np.float64(3.0): 6, np.float64(4.0): 10, \
        np.float64(5.0): 12, np.float64(6.0): 11, np.float64(7.0): 10, np.float64(8.0): 3, np.float64(9.0): 3, np.float64(10.0): 4})

        The scores are kind of well distributed majorly centered around mid range [4, 7] with some values in the extremes.
        Hence in this case it is clear that 5 becomes a good threshold. hence I am choosing <=5 to be low pain (35/66)
        '''
        test_set_matrix = np.concatenate((combined_matrix_data[0][0:2], combined_matrix_data[3][5:7]), axis = 0)
        
        #test_set_matrix = np.concatenate([combined_matrix_data[0], combined_matrix_data[3]], axis = 0)

        rem = np.concatenate((combined_matrix_data[0][3:],combined_matrix_data[3][0:5], combined_matrix_data[3][7:]), axis = 0)
        
        list_matrices = [rem] + combined_matrix_data[1:3] + combined_matrix_data[4:]
        train_set_matrices = np.concatenate(list_matrices, axis = 0)

        test_set_scores = np.concatenate((combined_scores[0][0:2], combined_scores[3][5:7]), axis = 0)
        
        rem_scores = np.concatenate((combined_scores[0][3:],combined_scores[3][0:5], combined_scores[3][7:]), axis = 0)
        
        rem_scores = [rem_scores] + combined_scores[1:3] + combined_scores[4:]
        train_set_scores = np.concatenate(rem_scores, axis = 0)


        train_set_matrices = train_set_matrices[:-1]
        train_set_scores = train_set_scores[:-1]
#--------------------------------------------------------------------------------
#------------------822e28-------------------------------------------------------
#822e28 : get [1][0:2] '0' as low pain and [2][-2:] '6' as high pain
    if sub_id=='822e28':
        '''
        #0 is a good threshold, 19/36 scores are 0 and clearly indicates nopain

        OrderedDict({np.float64(0.0): 19, np.float64(1.0): 1, np.float64(2.0): 1, np.float64(3.0): 6, np.float64(5.0): 5, np.float64(6.0): 4})
        Patient majorly felt no pain as per the scores, which means the other values except for 0 clearly indicates pain hence 0 is a good threshold
        >0 is pain
        '''
        test_set_matrix = np.concatenate((combined_matrix_data[1][0:2], combined_matrix_data[2][-2:]), axis = 0)
        
        #test_set_matrix = np.concatenate([combined_matrix_data[0], combined_matrix_data[3]], axis = 0)

        rem = np.concatenate((combined_matrix_data[1][2:],combined_matrix_data[2][0:5]), axis = 0)
        
        list_matrices = [rem] + [combined_matrix_data[0]] + combined_matrix_data[3:]
        train_set_matrices = np.concatenate(list_matrices, axis = 0)

        test_set_scores = np.concatenate((combined_scores[1][0:2], combined_scores[2][-2:]), axis = 0)
        
        rem_scores = np.concatenate((combined_scores[1][2:],combined_scores[2][0:5]), axis = 0)
        
        rem_scores = [rem_scores] + [combined_scores[0]] + combined_scores[3:]
        train_set_scores = np.concatenate(rem_scores, axis = 0)

#-------------------------------------------------------------------------------
#------------------422bc5-------------------------------------------------------
#422bc5 : get [1][-1] and [1][-3] '0' as low pain and [0][-2:] '6' as high pain
#Data is poorly distributed over the score range, not enough data in the low pain region. Hence performing poorly
    if sub_id == '422bc5':
        '''
        #the data is distributed from [2,8], OrderedDict({np.float64(2.0): 1, np.float64(3.0): 3, np.float64(4.0): 14, np.float64(5.0): 14, np.float64(7.0): 6, np.float64(8.0): 17})
        Patient has severly moderate high pain (8) as per the scores, also apart from this, the patient has felt more in the normal region (not pain, not no-pain) hence <=5 is nopain as per my suggestion
        # <=5 is low pain
        '''
        test_set_matrix = np.concatenate((combined_matrix_data[1][-1][np.newaxis, :, :], combined_matrix_data[1][-3][np.newaxis, :, :], combined_matrix_data[0][-2:]), axis = 0)
        
        #test_set_matrix = np.concatenate([combined_matrix_data[0], combined_matrix_data[3]], axis = 0)

        rem = np.concatenate((combined_matrix_data[1][0:9],combined_matrix_data[1][10][np.newaxis, :, :], combined_matrix_data[0][0:5]), axis = 0)
        
        list_matrices = [rem] + combined_matrix_data[2:]
        train_set_matrices = np.concatenate(list_matrices, axis = 0)

        test_set_scores = np.concatenate((combined_scores[1][-1][np.newaxis], combined_scores[1][-3][np.newaxis], combined_scores[0][-2:]), axis = 0)
        
        rem_scores = np.concatenate((combined_scores[1][0:9],combined_scores[1][10][np.newaxis],combined_scores[0][0:5]), axis = 0)
        
        rem_scores = [rem_scores]  + combined_scores[2:]
        train_set_scores = np.concatenate(rem_scores, axis = 0)

        # # breakpoint()
        train_set_matrices = train_set_matrices[:-1]
        train_set_scores = train_set_scores[:-1]


#-------------------------------------------------------------------------------

#------------------0b5a2e-------------------------------------------------------
#422bc5 : get [3][1:3] and [1][-3] '8' as high pain and [-1][1:] '3' as low pain
    if sub_id=='0b5a2e':
        '''
        OrderedDict({np.float64(2.0): 5, np.float64(3.0): 11, np.float64(4.0): 22, np.float64(5.0): 16, np.float64(6.0): 5, np.float64(7.0): 7, np.float64(8.0): 4})
        very less times the patient has felt extreme pains, most of the times, the patient has felt moderate pain [4 and 5]. 
        The model will obviously have hard time classifying it into high or low pain states, but might slightly do better in classifying low pain better
        because of the distribution (3 has good representation)
        Hence I am selecting the threshold to be >=5 as high pain
        
        '''
         
        test_set_matrix = np.concatenate((combined_matrix_data[3][1:3], combined_matrix_data[-1][1:]), axis = 0)
        
        #test_set_matrix = np.concatenate([combined_matrix_data[0], combined_matrix_data[3]], axis = 0)

        rem = np.concatenate((combined_matrix_data[3][0][np.newaxis, :, :],combined_matrix_data[3][3:], combined_matrix_data[-1][0][np.newaxis, :, :]), axis = 0)
        
        list_matrices = [rem] + combined_matrix_data[0:3] + combined_matrix_data[4:6]
        train_set_matrices = np.concatenate(list_matrices, axis = 0)

        test_set_scores = np.concatenate((combined_scores[3][1:3], combined_scores[-1][1:]), axis = 0)
        
        rem_scores = np.concatenate((combined_scores[3][0][np.newaxis],combined_scores[3][3:],combined_scores[-1][0][np.newaxis]), axis = 0)
        
        rem_scores = [rem_scores]  + combined_scores[0:3] + combined_scores[4:6]
        train_set_scores = np.concatenate(rem_scores, axis = 0)

        # breakpoint()
        train_set_matrices = [train_set_matrices[0:16]] + [train_set_matrices[18:]]
        train_set_matrices = np.concatenate(train_set_matrices, axis = 0)
        train_set_scores = [train_set_scores[0:16]] + [train_set_scores[18:]]
        train_set_scores = np.concatenate(train_set_scores, axis = 0)


#-------------------------------------------------------------------------------



    labels_test = pain_binary(test_set_scores)
    labels_train = pain_binary(train_set_scores)
    # breakpoint()
    #train_set_matrices (66, 129, 150000)
    #test_set_matrix (4, 129, 150000)

    trials_train = train_set_matrices.shape[0] #66
    trials_test = test_set_matrix.shape[0] #

    #observing only 2.5th minute
    # train_set_matrices = train_set_matrices[:, :, train_set_matrices.shape[-1]//2][:, :, np.newaxis]
    # test_set_matrix = test_set_matrix[:, :, test_set_matrix.shape[-1]//2][:, :, np.newaxis]
    # train_set_matrices = train_set_matrices[:, :, train_set_matrices.shape[-1]//2:]
    # test_set_matrix = test_set_matrix[:, :, test_set_matrix.shape[-1]//2:]
    # breakpoint()
    for j in range(trials_train):
        ecog_data= train_set_matrices[j][~np.all(train_set_matrices[j] == 0, axis = 1)] #getting non zero channels, always (array([  0, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]),)
        
        augmented_data=sliding_window_augmentation(ecog_data)
        for subarrays in augmented_data:
        # print(f'ecog data of trial {j}','\n',ecog_data,'\n','shape=',ecog_data.shape)
        # data=calc_features_bandpower(subarrays)
            #---------trial-------------------
            data = subarrays
            data = PIB(subarrays)
            if data.shape == (0,):
                breakpoint()
            if labels_train[j]==0:
                nopain_data_train.append(data)
            else:
                pain_data_train.append(data)
    # breakpoint()
    pain_data_train = np.array(pain_data_train)
    nopain_data_train = np.array(nopain_data_train)

    for j in range(trials_test):
        ecog_data= test_set_matrix[j][~np.all(test_set_matrix[j] == 0, axis = 1)] #getting non zero channels, always (array([  0, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]),)
        
        augmented_data=sliding_window_augmentation(ecog_data)
        for subarrays in augmented_data:
        # print(f'ecog data of trial {j}','\n',ecog_data,'\n','shape=',ecog_data.shape)
        # data=calc_features_bandpower(subarrays)
            #---------trial-------------------
            data = subarrays
            data = PIB(subarrays)
            if labels_test[j]==0:
                nopain_data_test.append(data)
            else:
                pain_data_test.append(data)
    # breakpoint()
    pain_data_test = np.array(pain_data_test)
    nopain_data_test = np.array(nopain_data_test)






    # breakpoint()
    return pain_data_train, nopain_data_train, pain_data_test, nopain_data_test

#The data needs to be balanced to apply csp filters

def calc_csp(pain_data_train, nopain_data_train, test_data):
    '''
    pain_data_train #[B, C, 6] #(900,90,6)
    nopain_data_train #[B, C, 6]
    pain_data_test #[B, C, 6]
    nopain_data_test #[B, C, 6]

    Function description : Calculates the spatial filter based on the pain and nopain data from the train set and 
                           applying it to both train set and the test set
    '''

    csp_filter = CSP(pain_data_train, nopain_data_train) #csp_filter[0] is pain, csp_filter[1] is nopain
    #csp_filter has shape (2, channels, components) 

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
     

if __name__ == '__main__':
    #file = load_dataset()
    # breakpoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", help = "Enter the subject id")
    args = parser.parse_args()
    sub_id = args.sub
    pain_data_train, nopain_data_train, pain_data_test, nopain_data_test = forward(sub_id)
    breakpoint()
    test_data = np.concatenate((pain_data_test, nopain_data_test), axis = 0)
    y_test=np.concatenate([np.ones(pain_data_test.shape[0]),np.zeros(nopain_data_test.shape[0])])
    y_train = np.concatenate([np.ones(pain_data_train.shape[0]),np.zeros(nopain_data_train.shape[0])])
    print("operation finished")
    # breakpoint()
    print("applying csp transformation and getting train and test sets....")
    train_set, test_set = calc_csp(pain_data_train, nopain_data_train, test_data)
    print("operation finished")
    #breakpoint()
    #X_train = train_set.copy()  #(1980,6,116)
    #X_test = test_set.copy()  #(60, 6, 116)
    train_set = train_set.reshape(train_set.shape[0],-1) #(1980, 696)
    test_set = test_set.reshape(test_set.shape[0],-1) #(60, 696)


    n_estimators_range = range(1, 100, 5)

    # Initialize lists to store training and validation accuracies
    train_scores_itr = []
    mean_scores_train = []
    mean_scores_val = []
    max_train=[]
    min_train=[]
    max_val=[]
    min_val=[]
    val_scores_itr = []
    num_itr=100
    # Train the model with different number of trees and record the accuracies
    for n_estimators in tqdm.tqdm(n_estimators_range):
      for n_iter in range(num_itr):

        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(train_set, y_train)

        train_pred = rf.predict(train_set)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_scores_itr.append(train_accuracy)

        val_pred = rf.predict(test_set)
        val_accuracy = accuracy_score(y_test, val_pred)
        val_scores_itr.append(val_accuracy)
      mean_scores_train.append(np.mean(train_scores_itr))
      mean_scores_val.append(np.mean(val_scores_itr))
      max_train.append(max(train_scores_itr))
      min_train.append(min(train_scores_itr))
      max_val.append(max(val_scores_itr))
      min_val.append(min(val_scores_itr))


    # Plot the training and validation accuracies
    breakpoint()
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, mean_scores_train, label=f'Mean Train Accuracy (Max : {np.max(mean_scores_train)})', color='blue',linewidth=3)
    plt.fill_between(n_estimators_range, min_train, max_train, alpha=0.3, color='blue')

    plt.plot(n_estimators_range, mean_scores_val, label=f'Mean Validation Accuracy (Max : {np.max(mean_scores_val)})', color='red',linewidth=3)
    plt.fill_between(n_estimators_range, min_val, max_val, alpha=0.3, color='red')

    plt.xlabel('Number of trees',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    #plt.ylim(min,1.1)
    
    plt.title('Random Forest Classifier Performance',fontsize=20)
    plt.legend(fontsize=16)
    plt.show()

    #plt.savefig(f"/home/remotelab/sid/Experiments/csp_filter_updated/{sub_id}_num_trees_trial.png")
    plt.close()
    ind = np.argmax(max_val)
    best_n_estimators = n_estimators_range[ind]
    rf_best=RandomForestClassifier(random_state=42)
    rf_best.fit(train_set,y_train)
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

    for sample_size_index in tqdm.tqdm(range(len(test_sample_sizes))):
        accuracy_sample_size = []
        precision_sample_size = []
        recall_sample_size = []

        for iter in range(num_iterations):
            indices = random.choices(range(len(test_set)), k=test_sample_sizes[sample_size_index])
            test_data_boot = test_set[indices]
            y_test_boot = y_test[indices]
            y_pred = rf_best.predict(test_data_boot)
            Y_PRED.append(y_pred)
            accuracy = accuracy_score(y_test_boot, y_pred)
            precision = precision_score(y_test_boot, y_pred)
            recall = recall_score(y_test_boot, y_pred)
            accuracy_sample_size.append(accuracy)
            precision_sample_size.append(precision)
            recall_sample_size.append(recall)
        ACC.append(accuracy_sample_size)
        acc.append(np.mean(accuracy_sample_size))
        std_acc.append(np.std(accuracy_sample_size))
        acc_max.append(max(accuracy_sample_size))
        acc_min.append(min(accuracy_sample_size))
        prec.append(np.mean(precision_sample_size))
        std_prec.append(np.std(precision_sample_size))
        rec.append(np.mean(recall_sample_size))
        std_rec.append(np.std(recall_sample_size))
    print(f"Maximum mean accuracy of {sub_id} is {np.max(acc)}")
    std_acc = np.array(std_acc)
    plt.plot(np.arange(0,len(acc)),acc, label=f'Mean accuracy (Max : {np.max(acc)})',linewidth=3)
    plt.fill_between(np.arange(0,len(acc)), acc-(std_acc/2), acc + (std_acc/2), alpha=0.3, color='blue')

    plt.title("Accuracy vs test samples", fontsize=20)
    plt.xlabel("Number of test samples", fontsize=20)
    plt.legend(loc='upper right', fontsize=14)
    plt.ylabel("Accuracy", fontsize=20)
    # plt.ylim(0,1.5)
    plt.grid("True")
    plt.show()
    # breakpoint()
    #plt.savefig(f"/home/remotelab/sid/Experiments/csp_filter_updated/{sub_id}_test.png")
    plt.savefig(f"/home/remotelab/sid/Experiments/Segmenting_2.5_mins/{sub_id}_test_post_2_5.png")
    # plt.savefig("0b5a2e_test.png")
    
    plt.close()






        