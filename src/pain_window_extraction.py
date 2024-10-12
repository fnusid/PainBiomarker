import numpy as np
import pandas as pd
import h5py
import tqdm
import datetime
import pdb
import shutil
import os
from collections import defaultdict

#TODO : The naming is not consistent in the xlsx files and the format of the date is also not consistent

def load_dataset_path():
    """
    Return a dictionary with keys as folder names and values as paths to the files (.h5)
    Input : None
    Output : Dictionary
    """
    path_dict = defaultdict()
    root = "/Volumes/MyPassport/Pain Data/"
    folders = os.listdir(root)
    # for folder in folders:
    #     path_dict[os.path.join(root, folder)] = []
    for folder in folders:
        folder_path = os.path.join(root, folder)
        files = os.listdir(folder_path)
        file_paths = [os.path.join(folder_path, filed) for filed in files]
        path_dict[folder_path] = file_paths
    # breakpoint()  
    return path_dict






if __name__ == '__main__':
    # --------------------------
    # VARS TO CHANGE
    outdir_root = "/Users/sidharth/Desktop/Herron_lab/spring_2024/data/extended_dataset"
    csv_files_root = "/Users/sidharth/Desktop/Herron_lab/spring_2024/data/VAS_timestamps"
    #create a load dataset function  
    dataset = load_dataset_path() #returns a dictionary with file paths
    sub_ids = [folder_names.split('/')[-1].split(' ')[0] for folder_names in dataset.keys()] #['0b5a2e', 'c5a5e9', '422bc5', '822e28', '6c29e3']
    # breakpoint()
    for idx in range(len(sub_ids)):
        sub_id = sub_ids[idx]
        if os.path.exists(os.path.join(outdir_root, sub_id))==False: 
            os.mkdir(os.path.join(outdir_root, sub_id))
        outdir_sub = os.path.join(outdir_root, sub_id)
        folder_path = [folder_names for folder_names in dataset.keys() if folder_names.split('/')[-1].split(' ')[0]==sub_id][0]
        for files in dataset[folder_path]:
            fn = files
            # breakpoint()
            if os.path.getsize(fn)<1024:
                if os.path.exists(os.path.join(outdir_sub, f"{sub_id}_{files.split('_')[-1]}"))==False:
                    shutil.copy(fn, os.path.join(outdir_sub, f"{sub_id}_{files.split('_')[-1]}"))
                continue
            fn_out = os.path.join(outdir_sub, f"{sub_id}_{files.split('_')[-1]}")
            if os.path.exists(fn_out):
                continue
            xlx_file = [x for x in os.listdir(csv_files_root) if x.startswith(sub_id)][0]
            try:
                csv = pd.read_excel(os.path.join(csv_files_root, xlx_file))
                
            except ValueError:
                breakpoint()

            

    # fn = "/Volumes/MyPassport/Pain Data/6c29e3 Neural/processed_6c29e3_1.h5"
    # fn_out = "/Volumes/MyPassport/Pain Data/6c29e3 Neural/6c29e3_d3_MPQ_5min.h5"
    # csv = pd.read_excel("/Users/sidharth/Desktop/Herron_lab/spring_2024/data/VAS_timestamps/6c29e3_Event_TS.xlsx")
    # breakpoint()
            date_timestamp_list = csv['time']

            # vars that are stable
            window_length = 5

            # prints
            print('file in: {}\nfile out:{}\ndatelist:{}\n'.format(fn, fn_out, date_timestamp_list))

            # --------------------------------

            fin = h5py.File(fn, 'r')

            t_start = fin.get('start_timestamp')[()]
            t_start_utc = datetime.datetime.utcfromtimestamp(int(t_start))
            data_ecog = fin.get('dataset')
            fs = int(fin.get('f_sample')[()])
            chan_label = fin.get('chanLabels')[()]
            chan_clean = chan_label.decode('utf-8').replace("'",'').replace(',','').replace('[','').replace(']','')
            chan_label = chan_clean.split()
            chs = chan_label[:198]

            # timestamp things
            total_time = len(data_ecog[0]) / fs
            t_end_utc = t_start_utc + datetime.timedelta(seconds = total_time)
            indices = []
            timestamps = []
            for idx in range(len(date_timestamp_list)):
                if date_timestamp_list[idx] <= t_end_utc:
                    timestamps.append(datetime.datetime.strptime(str(date_timestamp_list[idx]), '%Y-%m-%d %H:%M:%S'))
                    # indices.append(idx)

            #timestamps = [datetime.datetime.strptime(str(date_str), '%Y-%m-%d %H:%M:%S') for date_str in date_timestamp_list if date_str < t_end_utc]
            event_times = [x - t_start_utc for x in timestamps]
            # target_sample = [i.seconds * fs for i in event_times if i.days == 0]
            target_sample = []
            for idx in range(len(event_times)):
                if event_times[idx].days==0:
                    target_sample.append(event_times[idx].seconds * fs)
                    indices.append(idx)
            intensity_data = np.array(csv['intensity'][indices], dtype=np.float64)
            print('found {}/{} datetimes'.format(len(target_sample),len(event_times)))
            # breakpoint()
            window_length_s = window_length*60*fs #5 minute
            start_frame = [int(i - window_length_s/2) for i in target_sample]
            stop_frame = [int(i + window_length_s/2) for i in target_sample]
            if len(start_frame)==0 and len(stop_frame)==0:
                continue
            # extract windows   
            neural_windows = np.zeros((len(target_sample), len(chs), stop_frame[0] - start_frame[0]))
            
            for i in tqdm.tqdm(range(len(target_sample))):
                try:
                    neural_windows[i, :, :] = data_ecog[:len(chs), start_frame[i]:stop_frame[i]]
                except ValueError:
                    breakpoint()
                # print('finished target sample {}/3'.format(i+1))
            # breakpoint()
            data_out = h5py.File(fn_out, "w")
            data_out.create_dataset("neural_windows", data=neural_windows, chunks=True)
            data_out.create_dataset("fs", data=fs, dtype='i8')
            dt = h5py.special_dtype(vlen=str)
            data_out.create_dataset("channels", data=chs, dtype=dt)
            data_out.create_dataset("target_sample", data=target_sample, dtype = 'i')
            data_out.create_dataset("intensity", data = intensity_data)
            # breakpoint()
    print('finished saving data')