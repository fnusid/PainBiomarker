import numpy as np
import h5py as hp
import scipy.linalg as la
import torch
import pdb
import random
import sys
import torch.optim as optim
from sklearn.metrics import accuracy_score
import tqdm
import argparse
import scipy.signal as signal
import torch.nn as nn
import torch.nn.functional as F
from utils import load_dataset, sliding_window_augmentation, pain_binary
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


'''
 this script is separate for subject 822e28
 Personalizzed label binaralizationa and classification using LSTM
'''


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
    if components != 'full':
        
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
    # std_acc = np.array(std_acc)
    # plt.plot(np.arange(0,len(acc)),acc, label=f'Mean accuracy (Max : {np.max(acc)})',linewidth=3)
    # plt.fill_between(np.arange(0,len(acc)), acc-(std_acc/2), acc + (std_acc/2), alpha=0.3, color='blue')

    # plt.title("Accuracy vs test samples", fontsize=20)
    # plt.xlabel("Number of test samples", fontsize=20)
    # plt.legend(loc='upper right', fontsize=14)
    # plt.ylabel("Accuracy", fontsize=20)
    # # plt.ylim(0,1.5)
    # plt.grid("True")
    # plt.show()
    # # breakpoint()
    # #plt.savefig(f"/home/remotelab/sid/Experiments/csp_filter_updated/{sub_id}_test.png")
    # plt.savefig(f"/home/remotelab/sid/Experiments/csp_pib/c5a5e9/{sub_id}_test.png")
    # # plt.savefig("0b5a2e_test.png")
    
    # plt.close()
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
        
    combined_scores = np.concatenate(combined_scores)
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
        ignore_scores = [5,6,7]
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
        TOTAL_DATA.append(ecog_data)
        scores.append(labels[j])


    # for j in range(combined_matrix_data_selected.shape[0]):
    #     ecog_data = combined_matrix_data_selected[j][~np.all(combined_matrix_data_selected[j]==0, axis = 1)]
    #     augmented_data=sliding_window_augmentation(ecog_data)
    #     for subarrays in augmented_data:
    #         data = subarrays
    #         #data = PIB(subarrays)
    #         if data.shape[0] == 0:
    #             continue
    #         TOTAL_DATA.append(data)
    #         scores.append(labels[j])
    # breakpoint()
    TOTAL_DATA = np.array(TOTAL_DATA)
    scores = np.array(scores)
    # breakpoint()
    return TOTAL_DATA, scores

class lstm_classification(nn.Module):
    def __init__(self,input_dims):
        super().__init__()
        # pdb.set_trace()

        # Linear projection layer
        self.linear1 = nn.Linear(input_dims, 32)

        #use lstm to learn the data, use augmented 10s as time sequences
        # self.lstm = nn.LSTM(input_dims, hidden_dims, num_layers = 2)
        self.lstm = nn.LSTM(32, 32)

        #project to two dimensions
        # self.project = nn.Conv1d(hidden_dims, num_classes, kernel_size = 1)
        self.project = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pdb.set_trace()
        #breakpoint()
        x = nn.functional.relu(self.linear1(x))
        x = torch.unsqueeze(x, dim=1)   # temporal dim
        x, _ = self.lstm(x)
        x = x[-1, :, :] # get the last time step output of lstm
        x = self.project(x)
        # x = self.sigmoid(x)
        x = torch.squeeze(x)
        x = self.sigmoid(x)
        predictions = (x >= 0.5).int()
        # print(x)
        return x, predictions

def train_model(model, criterion, optimizer, train_set,y_train):
    model.train()
    all_preds = []
    all_labels = []
    total_loss = 0
    # y_train = y_train[:, 0]
    pos_l, neg_l = 0, 0
    # breakpoint()
    for idx in range(len(train_set)):
        inputs, labels = train_set[idx], y_train[idx][-1]
        #inputs = inputs.T
        if labels.all() == 0:
            neg_l += 1
        else:
            pos_l += 1
        optimizer.zero_grad()
        outputs, predictions = model(inputs)
        # if idx == 0:
        #     print("Model outputs : ", outputs)
        #     print("Target labels : ", labels)
        #breakpoint()
        all_preds.append(predictions)
        all_labels.append(labels)
        loss = criterion(outputs, labels.float())  # .squeeze() to match dimensions
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    # breakpoint()
    #print("Label balance train : ", neg_l, pos_l)
    accuracy = accuracy_score(all_labels, all_preds)
    return total_loss / len(train_set), accuracy

# Function to validate the model on one fold
def validate_model(model, criterion, test_set, y_test):
    # print("Inside validate model")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    pos_l, neg_l = 0, 0
    # y_test = y_test[:, 0]
    with torch.no_grad():   # Not necessary once model.eval() is set
        for idx in range(len(test_set)):
            inputs, labels = test_set[idx], y_test[idx][-1]
            #inputs = inputs.T

            if labels.all() == 0:
                neg_l += 1
            else:
                pos_l += 1

            outputs, predictions = model(inputs)
            loss = criterion(outputs, labels.float())
            total_loss += loss.item()
            all_preds.append(predictions)
            all_labels.append(labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    
    #print(" ######### Label balance val : ########## ", neg_l, pos_l)
    return total_loss / len(test_set), accuracy

def remove_outliers_iqr(epoch_data):
    """
    Removes outliers using the IQR method for a single epoch.
    """
    # Calculate Q1 and Q3 for the data of a given epoch
    Q1 = np.percentile(epoch_data, 25)
    Q3 = np.percentile(epoch_data, 75)
    IQR = Q3 - Q1

    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    filtered_data = epoch_data[(epoch_data >= lower_bound) & (epoch_data <= upper_bound)]
    return filtered_data

if __name__ == '__main__':
   
        parser = argparse.ArgumentParser()
        parser.add_argument("--sub", help = "Enter the subject id")

        args = parser.parse_args()
        sub_id = args.sub
        total_data, total_labels = forward(sub_id) #get total data 
        np.random.seed(42)
        #create a separate function here

        indices_sorted_labels =  np.argsort(total_labels)[::-1] #positive first
        total_data = total_data[indices_sorted_labels]
        total_labels = total_labels[indices_sorted_labels]
        # breakpoint()
        max_len_pos = np.max([x for x in range(len(total_labels)) if total_labels[x] == 1]) #generalize this
        positive_total_data = total_data[:max_len_pos+1]
        pos_labels = total_labels[:max_len_pos+1]
        negative_total_data = total_data[max_len_pos + 1 :]
        neg_labels = total_labels[max_len_pos+1:]
        n_folds = min(len(pos_labels), len(neg_labels))
        # breakpoint()
        mean_val_accuracies_fold = []
        mean_train_accuracies_fold = []
        val_acs_fold_ = []
        train_acs_fold_ =[] 
        for fold in range(n_folds):
            print("Fold ", fold+1)
            test_arr = []
            test_labels = []
            train_arr = []
            train_labels = []
            test_set_ind_pos = np.random.randint(max_len_pos+1)
            test_arr.append(positive_total_data[test_set_ind_pos])
            test_labels.append(pos_labels[test_set_ind_pos])
            train_arr.append(np.concatenate([positive_total_data[:test_set_ind_pos],positive_total_data[test_set_ind_pos+1:]], axis = 0))
            train_labels.append(np.concatenate([pos_labels[:test_set_ind_pos],pos_labels[test_set_ind_pos+1:]], axis = 0))
            train_neg_indices = np.random.choice(np.arange(len(neg_labels)), size = n_folds - 1, replace = False)
            train_arr.append(negative_total_data[train_neg_indices])
            train_labels.append(neg_labels[train_neg_indices])
            # breakpoint()
            test_arr.extend(negative_total_data[i] for i in range(len(neg_labels)) if i not in train_neg_indices)
            test_labels.extend(neg_labels[i] for i in range(len(neg_labels)) if i not in train_neg_indices)

            train_arr = np.array(train_arr)
            train_arr = train_arr.reshape(-1, train_arr.shape[-2], train_arr.shape[-1])
        
            test_arr = np.array(test_arr)
            train_labels = np.array(train_labels)
            train_labels = train_labels.reshape(-1)
            test_labels = np.array(test_labels)

            #slice into 10s epochs
            augmented_train_arr = []
            y_train = []

            print("Computing power in band for train set")
            for j in range(train_arr.shape[0]):
                ecog_data = train_arr[j][~np.all(train_arr[j]==0, axis = 1)]
                augmented_data=sliding_window_augmentation(ecog_data)
                for subarrays in augmented_data:
                    data = subarrays
                    data = PIB(subarrays)
                    if data.shape[0] == 0:
                        continue
                    augmented_train_arr.append(data)
                    y_train.append(train_labels[j])
           
            augmented_train_arr = np.array(augmented_train_arr)
            y_train = np.array(y_train)

            pos_indices = [ind for ind in range(len(y_train)) if y_train[ind] == 1]
            neg_indices = [ind for ind in range(len(y_train)) if y_train[ind] == 0]

            pain_data_train = augmented_train_arr[pos_indices]
            nopain_data_train = augmented_train_arr[neg_indices]
               
               
            augmented_test_arr = []
            y_test = []

            print("Computing power in band for test set")
            for j in range(test_arr.shape[0]):
                ecog_data = test_arr[j][~np.all(test_arr[j]==0, axis = 1)]
                augmented_data=sliding_window_augmentation(ecog_data)
                for subarrays in augmented_data:
                    data = subarrays
                    data = PIB(subarrays)
                    if data.shape[0] == 0:
                        continue
                    augmented_test_arr.append(data)
                    y_test.append(test_labels[j])
            augmented_test_arr = np.array(augmented_test_arr)
            y_test = np.array(y_test)
            # breakpoint()
            #get csp 
            train_set, test_set = calc_csp(pain_data_train, nopain_data_train, augmented_test_arr, components='full')
            
            train_set = torch.from_numpy(train_set.reshape(-1, 30, train_set.shape[-2] * train_set.shape[-1])).to(torch.float32)
            test_set = torch.from_numpy(test_set.reshape(-1, 30, test_set.shape[-2] * test_set.shape[-1])).to(torch.float32)
            # null hypothesis
            # np.random.shuffle(y_train)

            y_train = torch.from_numpy(y_train.reshape(-1, 30))
            y_test = torch.from_numpy(y_test.reshape(-1, 30))



    
            #pass it to lstm to classify
            num_epochs = 100
            input_dims = train_set.shape[-1] #later transposing the input
            model = lstm_classification(input_dims)

            criterion = nn.BCELoss()  # Use BCEWithLogitsLoss to avoid separate sigmoid
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            tr_accs = []
            val_accs = []
            # breakpoint()
            print("train set : ", train_set.shape)
            print("test set : ", test_set.shape)
            best_val = 0
            best_epoch = 0
            for epoch in range(num_epochs):
                train_loss, train_accuracy = train_model(model, criterion, optimizer, train_set, y_train)
                tr_accs.append(train_accuracy)
    
                print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | train_acc : {train_accuracy:.4f}')

                val_loss, val_accuracy = validate_model(model, criterion, test_set, y_test)
                val_accs.append(val_accuracy)
                print(f'Validation Loss: {val_loss:.4f} | val_acc : {val_accuracy:.4f}')
                if epoch>1:
                    if val_accuracy >= best_val:
                        best_val = val_accuracy
                        best_epoch = epoch
                        print(f"Best performance at {best_epoch+1}, val acc : {best_val}")
                        ckpt = {
                            'epoch' : epoch,
                            'model_state_dict' : model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                                }
                        torch.save(ckpt, f'/home/remotelab/sid/codebase/code/PainBiomarker/runs/model_weights/best_lstm_{sub_id}_fold{fold}.pth')

                # if train_accuracy < val_accuracy:
                #     break
            
            val_acs_fold_.append(val_accs)
            train_acs_fold_.append(tr_accs)
            model = lstm_classification(input_dims)
            checkpoint = torch.load(f'/home/remotelab/sid/codebase/code/PainBiomarker/runs/model_weights/best_lstm_{sub_id}_fold{fold}.pth')
            model.load_state_dict(checkpoint['model_state_dict'])

            _, valacc = validate_model(model, criterion, test_set, y_test)
            
            mean_val_accuracies_fold.append(np.mean(valacc))
            # mean_train_accuracies_fold.append(np.mean(tr_accs))

            # print(f"Maximum mean accuracy for subject {sub_id} for Fold {i} is {max_mean_acc}")
        # print(f"Mean train accuracy across all folds is {np.mean(mean_train_accuracies_fold)}")

        #plotting val accs of each epochs across folds
        # breakpoint()
        val_acs_fold_ = np.array(val_acs_fold_)
        train_acs_fold_ = np.array(train_acs_fold_)
        #calculate the interquartile range
        mean_vals = []
        std_vals = []
        mean_tr = []
        std_tr = []
        for epoch in range(val_acs_fold_.shape[1]):
            epoch_data = val_acs_fold_[:, epoch]  # Extract data for the current epoch across all folds
            filtered_data = remove_outliers_iqr(epoch_data)  # Remove outliers using IQR method

            # Calculate mean and standard deviation for the filtered data
            mean_vals.append(np.mean(filtered_data))
            std_vals.append(np.std(filtered_data))

        for epoch in range(train_acs_fold_.shape[1]):
            epoch_data = train_acs_fold_[:, epoch]  # Extract data for the current epoch across all folds
            filtered_data = remove_outliers_iqr(epoch_data)  # Remove outliers using IQR method

            # Calculate mean and standard deviation for the filtered data
            mean_tr.append(np.mean(filtered_data))
            std_tr.append(np.std(filtered_data))        
        breakpoint()
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)
        mean_tr = np.array(mean_tr)
        std_tr = np.array(std_tr)




        # Plotting
        epochs = np.arange(data.shape[1])  # 100 epochs

        plt.plot(epochs, mean_vals, label='Validation', color= 'orange')
        plt.plot(epochs, mean_tr, label='Train', color = 'blue')
        plt.fill_between(epochs, mean_vals - std_vals/2, mean_vals + std_vals/2, color='orange', alpha=0.3)
        plt.fill_between(epochs, mean_tr - std_tr/2, mean_tr + std_tr/2, color='blue', alpha=0.3)

        # plt.title('Mean Accuracy with Std Dev (Outliers Removed)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        plt.savefig(f"/home/remotelab/sid/codebase/code/PainBiomarker/runs/Experiments/PIB_CSP_LSTM/{sub_id}_lstm_pib_mean_all_epoch_mod.png")
    