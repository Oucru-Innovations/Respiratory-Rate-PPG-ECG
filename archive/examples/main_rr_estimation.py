import numpy as np
import pandas as pd
from RRest.estimate_rr import PKS, PZX, CtO, CtA, \
    Interpolate_RR, AR_RR, ARM, ARP, WCH

if __name__ == "__main__":
    files = ['segment-009.csv','segment-010.csv','segment-011.csv']
    rr_methods = [
        ARM,PKS,CtO,CtA,PZX,ARP,Interpolate_RR,WCH
    ]
    for fname in files:
        print(fname)
        df = pd.read_csv(f'../dataset/{fname}')
        sig = np.array(df['PLETH'])
        fs = 100
        for rr_method in rr_methods:
            resp = rr_method.get_rr(sig,fs)
            print(f"{rr_method.__name__}                 = {resp}")
