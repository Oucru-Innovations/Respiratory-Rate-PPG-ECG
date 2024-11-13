import importlib.util
import pandas as pd
import numpy as np
import sys
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import warnings
from fusion.fuse_rr import combine_estimations
from estimate_rr import AR_RR, ARM, ARP, CtA, CtO, EMD, freq_rr, PKS, PZX, WCH
from preprocess.preprocess import preprocess_signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

if __name__ == "__main__":

    calculated_fs = 256
    signal_type = 'ECG'
    ecg_data_path = "dataset/public_ecg_data.csv"
    ecg_target_path = "dataset/public_ecg_target.csv"
    
    # Load the ECG data and target files again, ensuring correct parsing
    ecg_data = pd.read_csv(ecg_data_path, header=None)
    ecg_data = preprocess_signal(ecg_data, calculated_fs, signal_type)
    ecg_target = pd.read_csv(ecg_target_path, header=None)

    target_rr = ecg_target.values.flatten()
    X, y = ecg_data, ecg_target.values.flatten()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=13)

    # Define and train a machine learning model
    model = xgb.XGBRegressor(n_estimators=50, random_state=11)

    # Hyper-parameter tuning using GridSearchCV
    param_grid_xgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.8, 0.9, 1.0],
        'tree_method': ['gpu_hist']
    }
    
    logger.info("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid_xgb, 
                            scoring='neg_mean_absolute_error', cv=5, n_jobs=2)
    grid_search.fit(X_train, y_train)
    
    best_models = {}
    best_models['xgb'] = grid_search.best_estimator_
    logger.info(f"Best parameters for XGB: {grid_search.best_params_}")
    logger.info(f"Best MAE for XGB: {abs(grid_search.best_score_)}")

    model = best_models['xgb']

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    logger.info(f"Training MAE: {train_mae}")
    logger.info(f"Test MAE: {test_mae}")
    logger.info(f"Predicted values (rounded) for test data: {np.round(y_pred_test[:200])}")
    logger.info(f"Actual values (rounded) for test data: {np.round(y_test.reshape(-1)[:200])}")
