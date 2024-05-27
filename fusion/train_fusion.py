import importlib.util
import pandas as pd
import numpy as np
import sys
sys.path.append(".")
from fusion.fuse_rr import combine_estimations
from estimate_rr import AR_RR, ARM, ARP, \
                        CtA, CtO, EMD, \
                        freq_rr, PKS, PZX, WCH
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,\
    BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import StackingRegressor
import xgboost as xgb

# Define file paths for the uploaded estimation methods
methods = {
    'AR_RR': AR_RR,
    'ARM': ARM,
    'ARP': ARP,
    'CtA': CtA,
    'CtO': CtO,
    'EMD': EMD,
    'freq_rr': freq_rr,
    'PKS': PKS,
    'PZX': PZX,
    'WCH': WCH
}

def get_estimations(signal, fs, signal_type='ECG', preprocess=True, fuse_method='voting', weights=None):
    estimations = []
    for name, method in methods.items():  
        rr_bpm = method.get_rr(signal, fs, preprocess, signal_type)
        estimations.append(rr_bpm)
    
    # combined_rr = combine_estimations(results, method=fuse_method, weights=weights)
    return estimations


if __name__ == "__main__":

    calculated_fs = 256
    signal_type = 'ECG'
    ecg_data_path = "dataset/public_ecg_data.csv"
    ecg_target_path = "dataset/public_ecg_target.csv"
    
    # calculated_fs = 100
    # signal_type = 'PPG'
    # ecg_data_path = "dataset/public_ppg_data.csv"
    # ecg_target_path = "dataset/public_ppg_target.csv"
    
    # Load the ECG data and target files again, ensuring correct parsing
    ecg_data = pd.read_csv(ecg_data_path, header=None)
    ecg_target = pd.read_csv(ecg_target_path, header=None)

    # Display the shape and first few rows of the data and target files
    # ecg_data_shape = ecg_data.shape
    # ecg_target_shape = ecg_target.shape

    # ecg_data_head = ecg_data.head()
    # ecg_target_head = ecg_target.head()
    
    target_rr = ecg_target.values.flatten()

    X = []
    y = []
    
    
    for index, signal in ecg_data.iterrows():
        estimations = get_estimations(signal.values, calculated_fs, 
                                      signal_type='ECG')
        X.append(estimations)
        y.append(ecg_target[index])
    
    
    X = np.array(X)
    y = np.array(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

    # Train a machine learning model
    model = RandomForestRegressor(n_estimators=100, criterion='absolute_error',
                                  random_state=7,max_features='log2')
    
    model = GradientBoostingRegressor(loss='huber',learning_rate=0.05, 
                                      n_estimators=200, random_state=7)
    model = BaggingRegressor(estimator=LinearRegression(), 
                             n_estimators=200, random_state=7)
    model = AdaBoostRegressor(estimator=LinearRegression(), 
                                n_estimators=100, random_state=42)
    model = xgb.XGBRegressor(n_estimators=50, random_state=11)
    
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"Training MAE: {train_mae}")
    print(f"Test MAE: {test_mae}")
    # print(np.round(y_pred_test[:200]))
    # print(np.round(y_test.reshape(-1)[:200]))
    
    # Define and train a stacking model
    base_learners = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=13)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=7)),
        ('bag', BaggingRegressor(estimator=RandomForestRegressor(), n_estimators=100, random_state=3)),
        ('ada', AdaBoostRegressor(estimator=LinearRegression(), n_estimators=100, random_state=2)),
        ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=11,learning_rate=0.5))
    ]

    stacking_model = StackingRegressor(estimators=base_learners, final_estimator=LinearRegression())
    stacking_model.fit(X_train, y_train)
    y_pred_train_stacking = stacking_model.predict(X_train)
    y_pred_test_stacking = stacking_model.predict(X_test)
    train_mae_stacking = mean_absolute_error(y_train, y_pred_train_stacking)
    test_mae_stacking = mean_absolute_error(y_test, y_pred_test_stacking)

    print(f"Stacking - Training MAE: {train_mae_stacking}, Test MAE: {test_mae_stacking}")
    print(np.round(y_pred_test_stacking[:200]))
    print(np.round(y_test.reshape(-1)[:200]))
    
    # # Apply the estimate_rr_peaks function to each segment
    # estimated_rr = []
    # for index, row in ecg_data.iterrows():
    #     segment = row.values
    #     rr = get_estimations(segment, calculated_fs,signal_type)
    #     estimated_rr.append(rr)

    # # Filter out None values from estimated_rr
    # valid_estimates = [(est, tgt) for est, tgt in zip(estimated_rr, target_rr) if est is not None]
    # estimated_rr_valid, target_rr_valid = zip(*valid_estimates)

    # # Convert to numpy arrays for easier comparison
    # estimated_rr_valid = np.array(estimated_rr_valid)
    # target_rr_valid = np.array(target_rr_valid)

    # # Calculate the Mean Absolute Error (MAE) as a simple metric of accuracy
    # mae = np.mean(np.abs(estimated_rr_valid - target_rr_valid))
    # print(mae)
    # print(np.round(estimated_rr_valid[:200]))
    # print( target_rr_valid[:200])  # Display MAE and first 10 estimated vs. target values for verification
