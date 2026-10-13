import pandas as pd
import numpy as np
from plotly import graph_objects as go
import os,sys
# sys.path.append(".")
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
import multiprocessing

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
    df = pd.read_csv(fname,
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
    df = pd.read_csv(fname,
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


def compose_train(PATH, SC_FILE,lock,total_samples = 10000):
    # # PATH = "dataset/24EIb-003-004/24EIb-003-004 31-05-2023" 
    # # PATH = "dataset/24EIb-003-005/24EIb-003-005 01-06-2023"
    # # PATH = "dataset/24EIb-003-005/24EIb-003-005 04-06-2023"
    # PATH = "dataset/24EIb-003-002/24EIb-003-002 26-05-2023"
    # SC_FILE = "3. PPG/SmartCareCsv_24EIb-002_26.05.2023.12.20.09_26.05.2023.18.21.12.csv"
    RR_FILE = "1. Monitor/NOM_RESPWaveExport.csv"
    PLETH_FILE = "1. Monitor/NOM_PLETHWaveExport.csv"
    ECG_FILE = "1. Monitor/NOM_ECG_ELEC_POTL_IIWaveExport.csv"
    META_FILE = "1. Monitor/MPDataExport.csv"
    # VIVALINK_FILE = "2. ECG/EcgCsv_24EIb-004_30.05.2023.18.18.15_31.05.2023.08.37.32.csv"

    df_rr = pd.read_csv(os.path.join(PATH,META_FILE),
                        usecols=['Time','SystemLocalTime','NOM_RESP_RATE'])
    
    # df_pleth = pd.read_csv(os.path.join(PATH,PLETH_FILE),
    #                     names=["date_machine","log_machine",
    #                             "date_laptop","value","unknow"],
    #                     header=None)

    df_sc = get_all_sc(os.path.join(PATH,SC_FILE))
    # df_vivalink = get_all_vivalink(os.path.join(PATH,VIVALINK_FILE))

    train_data = []
    train_target = []   

    for i in tqdm(range(total_samples)):
        idx = np.random.randint(1000,len(df_rr))
        # idx = 0
        time_event = dt.datetime.strptime(df_rr['Time'].iloc[idx],
                                            '%d-%m-%Y %H:%M:%S.%f')
        time_event_start = time_event - dt.timedelta(minutes=1)
        
        htd_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        time_event_start_sc_htd = htd_tz.localize(time_event_start - dt.timedelta(seconds=60))
        time_event_sc_htd = htd_tz.localize(time_event)
        
        # Convert the datetime object to timezone UTC
        time_event_start_sc_htd = time_event_start_sc_htd.astimezone(pytz.timezone('UTC'))
        time_event_sc_htd = time_event_sc_htd.astimezone(pytz.timezone('UTC'))

        # tz = pytz.timezone('UTC')
        # time_event_start_tz = tz.localize(time_event_start)
        # time_event_start_sc_tz = tz.localize(time_event_start_sc)
        # time_event_tz = tz.localize(time_event)
            
        # df_pleth['timestamp'] = pd.to_datetime(df_pleth['date_laptop'],
        #                                     format='%d-%m-%Y %H:%M:%S.%f')
        # mask_pleth = (df_pleth['timestamp'] >= time_event_start) \
        #         & (df_pleth['timestamp'] <=time_event)
        # subset_pleth = df_pleth.loc[mask_pleth]
        df_rr['timestamp'] = pd.to_datetime(df_rr['Time'],
                                            format='%d-%m-%Y %H:%M:%S.%f')
        mask_rr = (df_rr['timestamp'] >= time_event_start) \
                & (df_rr['timestamp'] <=time_event)
        subset_rr = df_rr.loc[mask_rr]
        
        # peaks = np.mean(subset_rr['NOM_RESP_RATE'])
        peaks = np.mean([int(i) for i in subset_rr['NOM_RESP_RATE'] if i != "-"])
        # peaks = signal.find_peaks_cwt(np.power(subset_rr['value'],3),widths=20)
        # if (len(peaks) <= 20) and (len(peaks)>=18):
        #     continue
        
        
        mask_sc = (df_sc['TIMESTAMP_MS'] >= time_event_start_sc_htd) \
            & (df_sc['TIMESTAMP_MS'] <=time_event_sc_htd)
        subset_sc = df_sc.loc[mask_sc]
            
        # mask_vivalink = (df_vivalink['timestamp'] >= time_event_start_sc_tz) \
        #     & (df_vivalink['timestamp'] <=time_event_tz)
        # subset_vivalink = df_vivalink.loc[mask_vivalink]
        
        if (len(subset_sc) > 0) and (not np.isnan(peaks)):    
            # subset = signal.resample(subset_sc['PLETH'].to_list(),100*30)
            # Only take 30s data for PPG
            subset = (subset_sc.iloc[len(subset_sc)-3000:])
            subset = subset['PLETH'].to_list()
            # fig = go.Figure()
            # # fig.add_trace(go.Scatter(x=np.arange(len(df_sc)),y=-(df_sc['PLETH']-33175)))
            # fig.add_trace(go.Scatter(x=np.arange(len(subset)),y=subset))
            # fig.show()
            
            # fig = go.Figure()
            # # fig.add_trace(go.Scatter(x=np.arange(len(df_sc)),y=-(df_sc['PLETH']-33175)))
            # fig.add_trace(go.Scatter(x=np.arange(len(subset_sc['PLETH'].to_list())),y=subset_sc['PLETH'].to_list()))
            # fig.show()
            
            # peaks = np.mean(subset_rr['NOM_RESP_RATE'])
            peaks = np.mean([int(i) for i in subset_rr['NOM_RESP_RATE'] if i != "-"])
            # peaks = signal.find_peaks_cwt(np.power(subset_rr['value'],3),widths=20)
            
            if len(subset)==3000:
                if (len(np.where(np.array(subset) != 0)[0]) == 3000):
                    train_data.append(subset)
                    train_target.append(int(peaks))
        # print(len(peaks))
        # row = []
        # row.append(len(np.array(subset_rr['value'])[peaks]))
    
    # queue.put((train_data,train_target))
    with lock:
        try:
            with open('dataset/ppg_train_data.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    train_data.append([float(x) for x in row])
            old_train_target = []

            with open('dataset/ppg_train_target.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    old_train_target.append([float(x) for x in row])
                train_target = train_target+old_train_target
            
            new_train_target = train_target[-1]
            train_target = train_target[:-1] + new_train_target
        except Exception as err:
            print(err)

        with open(os.path.join('dataset','ppg_train_data.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(train_data)
        with open(os.path.join('dataset','ppg_train_target.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(train_target)

#=======================================================================
#==============    VALIDATION SECTION ==================================
#=======================================================================

def compose_val(PATH, SC_FILE,lock,total_samples = 10000):
    # # PATH = "dataset/24EIb-003-004/24EIb-003-004 31-05-2023"
    # # PATH = "dataset/24EIb-003-005/24EIb-003-005 04-06-2023"
    # # PATH = "dataset/24EIb-003-005/24EIb-003-005 05-06-2023"
    # PATH = "dataset/24EIb-003-002/24EIb-003-002 27-05-2023"
    # SC_FILE = "3. PPG/SmartCareCsv_24EIb-002_27.05.2023.10.57.22_28.05.2023.04.20.18_lxelMqz.csv"
    RR_FILE = "1. Monitor/NOM_RESPWaveExport.csv"
    PLETH_FILE = "1. Monitor/NOM_PLETHWaveExport.csv"
    ECG_FILE = "1. Monitor/NOM_ECG_ELEC_POTL_IIWaveExport.csv"
    META_FILE = "1. Monitor/MPDataExport.csv"
    # VIVALINK_FILE = "2. ECG/EcgCsv_24EIb-003_27.05.2023.10.54.56_28.05.2023.08.11.40_QAsATgU.csv"

    df_rr = pd.read_csv(os.path.join(PATH,META_FILE),
                        usecols=['Time','SystemLocalTime','NOM_RESP_RATE'])
    
    # df_pleth = pd.read_csv(os.path.join(PATH,PLETH_FILE),
    #                     names=["date_machine","log_machine",
    #                             "date_laptop","value","unknow"],
    #                     header=None)

    df_sc = get_all_sc(os.path.join(PATH,SC_FILE))
    # df_vivalink = get_all_vivalink(os.path.join(PATH,VIVALINK_FILE))

    val_data = []
    val_target = []

    for i in tqdm(range(total_samples)):
        idx = np.random.randint(5000,len(df_rr))
        time_event = dt.datetime.strptime(df_rr['Time'].iloc[idx],
                                            '%d-%m-%Y %H:%M:%S.%f')
        time_event_start = time_event - dt.timedelta(minutes=1)
        
        htd_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        time_event_start_sc_htd = htd_tz.localize(time_event_start - dt.timedelta(seconds=50))
        time_event_sc_htd = htd_tz.localize(time_event)
        
        # Convert the datetime object to timezone UTC
        time_event_start_sc_htd = time_event_start_sc_htd.astimezone(pytz.timezone('UTC'))
        time_event_sc_htd = time_event_sc_htd.astimezone(pytz.timezone('UTC'))

        # tz = pytz.timezone('UTC')
        # time_event_start_tz = tz.localize(time_event_start)
        # time_event_start_sc_tz = tz.localize(time_event_start_sc)
        # time_event_tz = tz.localize(time_event)
            
        # df_pleth['timestamp'] = pd.to_datetime(df_pleth['date_laptop'],
        #                                     format='%d-%m-%Y %H:%M:%S.%f')
        # mask_pleth = (df_pleth['timestamp'] >= time_event_start) \
        #         & (df_pleth['timestamp'] <=time_event)
        # subset_pleth = df_pleth.loc[mask_pleth]
        df_rr['timestamp'] = pd.to_datetime(df_rr['Time'],
                                            format='%d-%m-%Y %H:%M:%S.%f')
        mask_rr = (df_rr['timestamp'] >= time_event_start) \
                & (df_rr['timestamp'] <=time_event)
        subset_rr = df_rr.loc[mask_rr]
        
        # resp = [int(i) for i in subset_rr['NOM_RESP_RATE'] if i != "-"]
        
        peaks = np.mean([int(i) for i in subset_rr['NOM_RESP_RATE'] if i != "-"])
        # if (len(peaks) <= 20) and (len(peaks)>=18):
        #     continue
        
        mask_sc = (df_sc['TIMESTAMP_MS'] >= time_event_start_sc_htd) \
            & (df_sc['TIMESTAMP_MS'] <=time_event_sc_htd)
        subset_sc = df_sc.loc[mask_sc]
            
        # mask_vivalink = (df_vivalink['timestamp'] >= time_event_start_sc_tz) \
        #     & (df_vivalink['timestamp'] <=time_event_tz)
        # subset_vivalink = df_vivalink.loc[mask_vivalink]
        
        if (len(subset_sc) > 0) and (not np.isnan(peaks)):    
            # subset = signal.resample(subset_sc['PLETH'].to_list(),100*30)
            subset = (subset_sc.iloc[len(subset_sc)-3000:])
            subset = subset['PLETH'].to_list()
            
            # peaks = np.mean(subset_rr['NOM_RESP_RATE'])
            peaks = np.mean([int(i) for i in subset_rr['NOM_RESP_RATE'] if i != "-"])
            
            if len(subset)==3000:
                if (len(np.where(np.array(subset) >0)[0]) == 3000):
                    val_data.append(subset)
                    val_target.append(int(peaks))

    # queue.put((val_data,val_target))
    with lock:
        try:
            with open('dataset/ppg_val_data.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    val_data.append([float(x) for x in row])
            old_val_target = []

            with open('dataset/ppg_val_target.csv', mode='r') as file:
                reader = csv.reader(file)
                for row in reader:
                    old_val_target.append([float(x) for x in row])
                val_target = val_target+old_val_target
                
            new_val_target = val_target[-1]
            val_target = val_target[:-1] + new_val_target
        except Exception as err:
            print(err)
            
        with open(os.path.join('dataset','ppg_val_data.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(val_data)
        with open(os.path.join('dataset','ppg_val_target.csv'), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(val_target)
    

if __name__ == '__main__':
    lock = multiprocessing.Lock()

    for i in tqdm(range(10)):       
        path_train_list = [
            "dataset/24EIb-003-001/24EIb-003-001 10-05-2023",
            "dataset/24EIb-003-002/24EIb-003-002 25-05-2023",
            "dataset/24EIb-003-003/24EIb-003-003 28-05-2023",
            "dataset/24EIb-003-004/24EIb-003-004 31-05-2023",
            "dataset/24EIb-003-005/24EIb-003-005 01-06-2023",
            "dataset/24EIb-003-006/24EIb-003-006 10-06-2023"
        ]
        ppg_train_list = [
            "3. PPG/SmartCareCsv_24EIb-001_10.05.2023.11.05.47_10.05.2023.13.01.48.csv",
            "3. PPG/SmartCareCsv_24EIb-002_25.05.2023.16.16.39_26.05.2023.08.35.06.csv",
            "3. PPG/SmartCareCsv_24EIb-003_28.05.2023.11.49.43_29.05.2023.09.16.17.csv",
            "3. PPG/SmartCareCsv_24EIb-004_31.05.2023.10.52.04_31.05.2023.18.03.10.csv",
            "3. PPG/SmartCareCsv_24EIb-005_01.06.2023.10.39.20_01.06.2023.22.08.25.csv",
            "3. PPG/SmartCareCsv_24EIb-006_10.06.2023.10.50.17_10.06.2023.21.16.58.csv"
        ]
        path_val_list = [
            "dataset/24EIb-003-001/24EIb-003-001 10-05-2023",
            "dataset/24EIb-003-002/24EIb-003-002 27-05-2023",
            "dataset/24EIb-003-003/24EIb-003-003 29-05-2023",
            "dataset/24EIb-003-004/24EIb-003-004 31-05-2023",
            "dataset/24EIb-003-005/24EIb-003-005 04-06-2023",
            "dataset/24EIb-003-006/24EIb-003-006 12-06-2023"
        ]
        ppg_val_list = [
            "3. PPG/SmartCareCsv_24EIb-001_10.05.2023.11.05.47_10.05.2023.13.01.48.csv",
            "3. PPG/SmartCareCsv_24EIb-002_27.05.2023.10.57.22_28.05.2023.04.20.18_lxelMqz.csv",
            "3. PPG/SmartCareCsv_24EIb-003_29.05.2023.14.34.11_29.05.2023.22.22.54.csv",
            "3. PPG/SmartCareCsv_24EIb-004_31.05.2023.10.52.04_31.05.2023.18.03.10.csv",
            "3. PPG/SmartCareCsv_24EIb-005_04.06.2023.11.03.16_04.06.2023.22.23.52.csv",
            "3. PPG/SmartCareCsv_24EIb-006_12.06.2023.11.50.11_13.06.2023.02.02.55.csv"
        ]
        
        # compose_train(path_train_list[0],ppg_train_list[0],lock=lock)
        
        process_train_list = [ 
            multiprocessing.Process(target=compose_train,args=(path_train_list[i],
                                                               ppg_train_list[i],
                                                               lock,
                                                               ))
            for i in range(len(ppg_train_list))
            # for i in range(2)
        ]
        
        process_val_list = [ 
            multiprocessing.Process(target=compose_val,args=(path_val_list[i],
                                                            ppg_val_list[i],
                                                            lock,
                                                               ))
            for i in range(6)
        ]
        
        # for p in process_train_list:
        #     p.start()
        
        for p in process_val_list:
            p.start()
        
        # for p in process_train_list:
        #     p.join()
        
        for p in process_val_list:
            p.join()
 
        print("All processes have completed")