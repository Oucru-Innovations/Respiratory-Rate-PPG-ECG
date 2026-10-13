import csv
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
import csv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datetime as dt
from efficientnet_pytorch import EfficientNet
import sys
sys.path.append("..")
import seaborn as sns
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
import mlflow

PATH = "/mnt/Data/Workspace/Benchmark/dataset"

def get_flat(x,flat):
    flat +=  ast.literal_eval(x)

def get_flat_single(x,flat):
    flat +=  ([x]*75)


def get_flat_ecg(x,flat,samples):
    values = ast.literal_eval(x)
    flat +=  values
    samples+=[len(values)]

# def get_flat_single_ecg(x,flat):
#     flat +=  ([x]*100)

def get_datetime(x,flat):
    try:
        start_time_converted = dt.datetime.strptime(x,format)
    except Exception as err:
        if type(x) is str:
            start_time_converted = dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S%z")
        else:
            start_time_converted = x
    flat.append(start_time_converted)

def get_flat_timestamp(x,flat):
    format = "%Y-%m-%d %H:%M:%S.%f%z"
    try:
        start_time_converted = dt.datetime.strptime(x,format)
    except Exception as err:
        if type(x) is str:
            start_time_converted = dt.datetime.strptime(x,"%Y-%m-%d %H:%M:%S%z")
        else:
            start_time_converted = x
    dt_list = [start_time_converted + dt.timedelta(milliseconds=i*10) for i in range(0,75)]
    flat += (dt_list)


def get_all_nonin(fname):
    df = pd.read_csv(fname,
                       usecols=["timestamp",
                                "o2","hr","pleth",
                                
                                ],
                       header=0)
    dict_df = {}
    pleth_data = []
    spo2_data = []
    hr_data = []
    # red_data = []
    # ir_data = []
    dt_data = []
    df['pleth'].apply(get_flat,flat=pleth_data)
    df['o2'].apply(get_flat_single, flat=spo2_data)
    df['hr'].apply(get_flat_single, flat=hr_data)
    # df['red'].apply(get_flat, flat=red_data)
    # df['ir'].apply(get_flat, flat=ir_data)
    df['timestamp'].apply(get_flat_timestamp, flat=dt_data)
    # dict_df['Timestamps'] = vs.generate_timestamp(start_datetime=start_time_converted,
    #                                   sampling_rate=100,signal_length=len(pleth_data))]
    dict_df['TIMESTAMP_MS'] = dt_data
    dict_df['PLETH'] = pleth_data
    dict_df['SPO2'] = spo2_data
    dict_df['HR'] = hr_data
    # dict_df['RED'] = red_data
    # dict_df['IR'] = ir_data
    
    
    df_new = pd.DataFrame(dict_df)
    # print(df_new())
    
    return df_new

def get_all_vivalink(fname):
    df = pd.read_csv(os.path.join(PATH,fname),
                        usecols=["timestamp","ecg","av_rr"],
                        header=0)
    dict_df = {}
    ecg_data = []
    sample_lens = []
    av_rr = []
    dt_data = []
    df['ecg'].apply(get_flat_ecg,flat=ecg_data,samples = sample_lens)
    
    for i in range(len(sample_lens)):
        
        av_rr +=  ([df['av_rr'].iloc[i]]*sample_lens[i])
        
        format = "%Y-%m-%d %H:%M:%S.%f%z"
        try:
            start_time_converted = dt.datetime.strptime(df['timestamp'].iloc[i],format)
        except Exception as err:
            if type(df['timestamp'].iloc[i]) is str:
                start_time_converted = dt.datetime.strptime(df['timestamp'].iloc[i],"%Y-%m-%d %H:%M:%S%z")
            else:
                start_time_converted = df['timestamp'].iloc[i]
        millisecond_delta = 1000/sample_lens[i]
        dt_list = [start_time_converted + dt.timedelta(milliseconds=millisecond_delta) for i in range(0,sample_lens[i])]
        # flat += (dt_list)
        dt_data+=dt_list       
        
    # df['av_rr'].apply(get_flat_single, flat=av_rr)
    # df['timestamp'].apply(get_flat_timestamp, flat=dt_data)
    # dict_df['Timestamps'] = vs.generate_timestamp(start_datetime=start_time_converted,
    #                                   sampling_rate=100,signal_length=len(pleth_data))]
    dict_df['timestamp'] = dt_data
    dict_df['ecg'] = ecg_data
    dict_df['av_rr'] = av_rr
    
    return pd.DataFrame(dict_df)
    # print(df_new())
    # return dict_df

# df_sc = get_all_nonin(os.path.join(PATH,"NoninCsv_66DX-4004_14.06.2023.07.07.07_14.06.2023.08.00.14.csv"))
# get_date_time(patient,fn)
# df_all = get_date_time(SC_FILE)


path = "/home/khoaldv/Downloads/holter-data/Holter ECG"
from scipy.io import loadmat

import wfdb
import struct
# from pyecg import ECGRecord

# rec = ECGRecord.from_ishine(os.path.join(path,"0000002.ecg"))
# print(rec)

# from ishneholterlib import Holter
# rec = Holter(os.path.join(path,"0000002.ecg"))
# rec = wfdb.rdrecord(os.path.join(path,"0000002.ecg"))
from tqdm import tqdm
ecg_arr = []

with open(os.path.join(path,"0000002.ecg"),'rb') as dat_file:
    i = 0
    for line in tqdm(dat_file):
        i = i+1
        if len(line) % 2!=0:
            line = line[:-1]
        
        decoded_value = struct.unpack('h' * (len(line)//2),line)
        ecg_arr = ecg_arr + list(decoded_value)
        # print(decoded_value)
        # print('======')
        if (i% 5000) == 0:
            print("save_temp_data")
            # np.savetxt('holter_ecg.qrs',np.array(ecg_arr),fmt='%.2f',delimiter=',')            
np.savetxt('holter_ecg.ecg',np.array(ecg_arr),fmt='%.2f',delimiter=',')