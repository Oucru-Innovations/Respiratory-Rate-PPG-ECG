import pandas as pd
import numpy as np
from plotly import graph_objects as go
import os,sys
from scipy import signal
import datetime as dt
import ast
from estimate_rr import CtA,CtO,ARM,AR_RR,ARP,freq_rr,\
    Interpolate_RR,PKS,PZX,time_rr,WCH
import csv
from tqdm import tqdm
import pytz
from scipy import signal
import csv

# PATH = "dataset/24EIb-003-101/24EIb-003-101 12-05-2023" #
# PATH = "dataset/24EIb-003-006/24EIb-003-006 03-06-2023"
# PATH = "dataset/24EIb-003-005/24EIb-003-005 18-05-2023"
# PATH = "dataset/24EIb-003-102/24EIb-003-102 20-05-2023"
PATH = "dataset/24EIb-003-004/24EIb-003-004 30-05-2023"

RR_FILE = "1. Monitor/NOM_RESPWaveExport.csv"
PLETH_FILE = "1. Monitor/NOM_PLETHWaveExport.csv"
ECG_FILE = "1. Monitor/NOM_ECG_ELEC_POTL_IIWaveExport.csv"
SC_FILE = "3. PPG/SmartCareCsv_24EIb-004_30.05.2023.18.18.14_31.05.2023.08.31.01.csv"
VIVALINK_FILE = "2. ECG/EcgCsv_24EIb-004_30.05.2023.18.18.15_31.05.2023.08.37.32.csv"

def get_flat(x,flat):
    flat +=  ast.literal_eval(x)

def get_flat_single(x,flat):
    flat +=  ([x]*100)

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
    dt_list = [start_time_converted + dt.timedelta(milliseconds=i*10) for i in range(0,100)]
    flat += (dt_list)

def get_all_sc(fname):
    df = pd.read_csv(os.path.join(PATH,fname),
                       usecols=["timestamp",
                                "o2","hr","pleth",
                                "red","ir"
                                ],
                       header=0)
    dict_df = {}
    pleth_data = []
    spo2_data = []
    hr_data = []
    red_data = []
    ir_data = []
    dt_data = []
    df['pleth'].apply(get_flat,flat=pleth_data)
    df['o2'].apply(get_flat_single, flat=spo2_data)
    df['hr'].apply(get_flat_single, flat=hr_data)
    df['red'].apply(get_flat, flat=red_data)
    df['ir'].apply(get_flat, flat=ir_data)
    df['timestamp'].apply(get_flat_timestamp, flat=dt_data)
    # dict_df['Timestamps'] = vs.generate_timestamp(start_datetime=start_time_converted,
    #                                   sampling_rate=100,signal_length=len(pleth_data))]
    dict_df['TIMESTAMP_MS'] = dt_data
    dict_df['PLETH'] = pleth_data
    dict_df['SPO2'] = spo2_data
    dict_df['HR'] = hr_data
    dict_df['RED'] = red_data
    dict_df['IR'] = ir_data
    
    
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

df_rr = pd.read_csv(os.path.join(PATH,RR_FILE),
                       names=["date_machine","log_machine",
                              "date_laptop","value","unknow"],
                       header=None)
df_pleth = pd.read_csv(os.path.join(PATH,PLETH_FILE),
                       names=["date_machine","log_machine",
                              "date_laptop","value","unknow"],
                       header=None)

df_sc = get_all_sc(SC_FILE)
df_vivalink = get_all_vivalink(VIVALINK_FILE)

train_data = []
train_target = []

total_samples = 10000

for i in tqdm(range(total_samples)):
    idx = np.random.randint(10000,len(df_rr))
    time_event = dt.datetime.strptime(df_rr['date_laptop'].iloc[idx],
                                        '%d-%m-%Y %H:%M:%S.%f')
    time_event_start = time_event - dt.timedelta(minutes=1)
    time_event_start_sc = time_event - dt.timedelta(minutes=1)
    tz = pytz.timezone('UTC')
    time_event_start_tz = tz.localize(time_event_start)
    time_event_start_sc_tz = tz.localize(time_event_start_sc)
    time_event_tz = tz.localize(time_event)
        
    # df_pleth['timestamp'] = pd.to_datetime(df_pleth['date_laptop'],
    #                                     format='%d-%m-%Y %H:%M:%S.%f')
    # mask_pleth = (df_pleth['timestamp'] >= time_event_start) \
    #         & (df_pleth['timestamp'] <=time_event)
    # subset_pleth = df_pleth.loc[mask_pleth]
    df_rr['timestamp'] = pd.to_datetime(df_rr['date_laptop'],
                                        format='%d-%m-%Y %H:%M:%S.%f')
    mask_rr = (df_rr['timestamp'] >= time_event_start) \
            & (df_rr['timestamp'] <=time_event)
    subset_rr = df_rr.loc[mask_rr]
        
    # mask_sc = (df_sc['TIMESTAMP_MS'] >= time_event_start_sc_tz) \
    #     & (df_sc['TIMESTAMP_MS'] <=time_event_tz)
    # subset_sc = df_sc.loc[mask_sc]
        
    mask_vivalink = (df_vivalink['timestamp'] >= time_event_start_sc_tz) \
        & (df_vivalink['timestamp'] <=time_event_tz)
    subset_vivalink = df_vivalink.loc[mask_vivalink]
    
    if len(subset_vivalink) > 0:    
        subset = signal.resample(subset_vivalink['ecg'].to_list(),128*60)
        train_data.append(subset)
        peaks = signal.find_peaks_cwt(np.power(subset_rr['value'],3),widths=20)
        train_target.append(len(peaks))
    # print(len(peaks))
    # row = []
    # row.append(len(np.array(subset_rr['value'])[peaks]))
    

#=======================================================================
#==============    VALIDATION SECTION ==================================
#=======================================================================

# PATH = "dataset/24EIb-003-006/24EIb-003-006 03-06-2023"
# PATH = "dataset/24EIb-003-101/24EIb-003-101 12-05-2023"
# PATH = "dataset/24EIb-003-005/24EIb-003-005 18-05-2023"
# PATH = "dataset/24EIb-003-102/24EIb-003-102 20-05-2023"
PATH = "dataset/24EIb-003-003/24EIb-003-003 27-05-2023"

RR_FILE = "1. Monitor/NOM_RESPWaveExport.csv"
PLETH_FILE = "1. Monitor/NOM_PLETHWaveExport.csv"
ECG_FILE = "1. Monitor/NOM_ECG_ELEC_POTL_IIWaveExport.csv"
SC_FILE = "3. PPG/SmartCareCsv_24EIb-003_27.05.2023.10.54.57_28.05.2023.04.15.37_k7qt1ML.csv"
VIVALINK_FILE = "2. ECG/EcgCsv_24EIb-003_27.05.2023.10.54.56_28.05.2023.08.11.40_QAsATgU.csv"

df_rr = pd.read_csv(os.path.join(PATH,RR_FILE),
                       names=["date_machine","log_machine",
                              "date_laptop","value","unknow"],
                       header=None)
df_pleth = pd.read_csv(os.path.join(PATH,PLETH_FILE),
                       names=["date_machine","log_machine",
                              "date_laptop","value","unknow"],
                       header=None)

df_sc = get_all_sc(SC_FILE)
df_vivalink = get_all_vivalink(VIVALINK_FILE)

train_data = []
train_target = []

total_samples = 10000

val_data = []
val_target = []

for i in tqdm(range(total_samples)):
    idx = np.random.randint(10000,len(df_rr))
    time_event = dt.datetime.strptime(df_rr['date_laptop'].iloc[idx],
                                        '%d-%m-%Y %H:%M:%S.%f')
    time_event_start = time_event - dt.timedelta(minutes=1)
    time_event_start_sc = time_event - dt.timedelta(minutes=1)
    tz = pytz.timezone('UTC')
    time_event_start_tz = tz.localize(time_event_start)
    time_event_start_sc_tz = tz.localize(time_event_start_sc)
    time_event_tz = tz.localize(time_event)
        
    # df_pleth['timestamp'] = pd.to_datetime(df_pleth['date_laptop'],
    #                                     format='%d-%m-%Y %H:%M:%S.%f')
    # mask_pleth = (df_pleth['timestamp'] >= time_event_start) \
    #         & (df_pleth['timestamp'] <=time_event)
    # subset_pleth = df_pleth.loc[mask_pleth]
    df_rr['timestamp'] = pd.to_datetime(df_rr['date_laptop'],
                                        format='%d-%m-%Y %H:%M:%S.%f')
    mask_rr = (df_rr['timestamp'] >= time_event_start) \
            & (df_rr['timestamp'] <=time_event)
    subset_rr = df_rr.loc[mask_rr]
        
    # mask_sc = (df_sc['TIMESTAMP_MS'] >= time_event_start_sc_tz) \
    #     & (df_sc['TIMESTAMP_MS'] <=time_event_tz)
    # subset_sc = df_sc.loc[mask_sc]
        
    mask_vivalink = (df_vivalink['timestamp'] >= time_event_start_sc_tz) \
        & (df_vivalink['timestamp'] <=time_event_tz)
    subset_vivalink = df_vivalink.loc[mask_vivalink]
    
    if len(subset_vivalink) > 0:    
        subset = signal.resample(subset_vivalink['ecg'].to_list(),128*60)
        val_data.append(subset)
        peaks = signal.find_peaks_cwt(np.power(subset_rr['value'],3),widths=20)
        val_target.append(len(peaks))

with open('dataset/train_data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        train_data.append([float(x) for x in row])
old_train_target = []

with open('dataset/train_target.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        old_train_target.append([float(x) for x in row])
    train_target = train_target+old_train_target
    
new_train_target = train_target[-1]
train_target = train_target[:-1] + new_train_target

with open(os.path.join('dataset','train_data.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_data)
with open(os.path.join('dataset','train_target.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(train_target)
    
#=====================================================

with open('dataset/val_data.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        val_data.append([float(x) for x in row])
old_val_target = []

with open('dataset/val_target.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        old_val_target.append([float(x) for x in row])
    val_target = val_target+old_val_target
    
new_val_target = val_target[-1]
val_target = val_target[:-1] + new_val_target

with open(os.path.join('dataset','val_data.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(val_data)
with open(os.path.join('dataset','val_target.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(val_target)
    
