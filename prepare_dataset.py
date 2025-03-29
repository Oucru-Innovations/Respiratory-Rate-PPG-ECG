import pandas as pd
import ast
import numpy as np
def process_and_combine_pleth(file_path, output_path):
    try:
        # Read CSV file safely
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

        # Convert timestamp to datetime, handling optional milliseconds, and timezone GMT+7
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, format='mixed').dt.tz_convert('Asia/Ho_Chi_Minh')

        # Safely evaluate the pleth string to list
        def safe_eval(x):
            try:
                return ast.literal_eval(x) if isinstance(x, str) else x
            except:
                return None

        df['pleth'] = df['pleth'].apply(safe_eval)
        df.dropna(subset=['pleth'], inplace=True)

        combined_segments = []

        for i in range(0, len(df), 60):
            chunk = df.iloc[i:i+60]
            if len(chunk) == 60:
                combined_pleth = [value for pleth_list in chunk['pleth'] for value in pleth_list]
                combined_segments.append({
                    'start_timestamp': chunk['timestamp'].iloc[0],
                    'end_timestamp': chunk['timestamp'].iloc[-1],
                    'pleth': combined_pleth
                })

        combined_df = pd.DataFrame(combined_segments)

        # Save to CSV
        combined_df.to_csv(output_path, index=False)

        print(f"Successfully processed data saved to '{output_path}'.")
        print(f"Total segments created: {len(combined_df)}.")

        return combined_df

    except Exception as e:
        print(f"Error occurred: {e}")
        return None

def map_closest_resp_rate(processed_df_path, comparison_df_path, output_path):
    try:
        # Load processed dataframe
        processed_df = pd.read_csv(processed_df_path)
        processed_df['end_timestamp'] = pd.to_datetime(processed_df['end_timestamp'], utc=True).dt.tz_convert('Asia/Ho_Chi_Minh')

        # Load comparison dataframe, handle mixed formats and timezone
        comparison_df = pd.read_csv(comparison_df_path, low_memory=False)

        # Handle timestamps with or without milliseconds, assuming local time GMT+7
        comparison_df['SystemLocalTime'] = pd.to_datetime(
            comparison_df['SystemLocalTime'], 
            format='mixed'
        ).dt.tz_localize('Asia/Ho_Chi_Minh', ambiguous='NaT', nonexistent='shift_forward')

        comparison_df.dropna(subset=['SystemLocalTime'], inplace=True)

        # Sort comparison dataframe for faster searching
        comparison_df.sort_values('SystemLocalTime', inplace=True)
        comparison_df.reset_index(drop=True, inplace=True)

        # Efficient closest timestamp search
        def find_closest_resp_rate(target_time):
            idx = comparison_df['SystemLocalTime'].searchsorted(target_time)
            if idx == len(comparison_df):
                idx -= 1
            elif idx > 0:
                prev_idx, next_idx = idx - 1, idx
                if abs((comparison_df.loc[prev_idx, 'SystemLocalTime'] - target_time)) < abs((comparison_df.loc[next_idx, 'SystemLocalTime'] - target_time)):
                    idx = prev_idx
            return comparison_df.loc[idx, 'NOM_RESP_RATE']

        # Map closest respiratory rate
        processed_df['NOM_RESP_RATE'] = processed_df['end_timestamp'].apply(find_closest_resp_rate)

        # Save final dataframe
        processed_df.to_csv(output_path, index=False)

        print(f"Respiratory rates mapped successfully. Output saved to '{output_path}'.")
        print(processed_df.head())

        return processed_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_data():
    input_path = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp/24EIb-003-029/23092023/PPG/SmartCareCsv_24EIb-029_23.09.2023.10.00.12_23.09.2023.22.21.01.csv"
    processed_path = "processed_pleth_segments.csv"

    # First step: process pleth data
    result_df = process_and_combine_pleth(input_path, processed_path)

    if result_df is not None:
        print("Preview of processed data:")
        print(result_df.head())
    else:
        print("Pleth signal processing failed.")
        return

    # Second step: map respiratory rate
    comparison_df_path = "/media/data/Workspace/Respiratory-Rate-PPG-ECG/dataset/Resp/24EIb-003-029/23092023/Monitor/MonitorPhillips-MPDataExport_24EIb-003-029_23.09.2023.10.04.34_24.09.2023.07.57.56.csv"
    output_path = "final_mapped_resp_rate.csv"

    mapped_df = map_closest_resp_rate(processed_path, comparison_df_path, output_path)

    if mapped_df is None:
        print("Mapping respiratory rate failed.")
    else:
        print("Respiratory rate mapping complete. Preview:")
        print(mapped_df.head())

def create_training_data(mapped_df_path, pleth_output_path, target_output_path):
    try:
        # Load final mapped dataframe
        mapped_df = pd.read_csv(mapped_df_path)

        # Convert pleth string representation back to numpy arrays
        pleth_data = mapped_df['pleth'].apply(lambda x: np.array(ast.literal_eval(x)))

        # Stack pleth data into 2D numpy array (samples x segment_length)
        train_data = np.vstack(pleth_data.values)

        # Extract NOM_RESP_RATE as a numpy array with shape (samples x 1)
        train_target = mapped_df['NOM_RESP_RATE'].values.reshape(-1, 1)

        # Save to CSV with numeric format (avoid scientific notation)
        np.savetxt(pleth_output_path, train_data, delimiter=",", fmt='%.5f')
        np.savetxt(target_output_path, train_target, delimiter=",", fmt='%d')

        print(f"Train data shape: {train_data.shape}")
        print(f"Train target shape: {train_target.shape}")
        print(f"Successfully saved training data to '{pleth_output_path}' and target data to '{target_output_path}'.")

    except Exception as e:
        print(f"Error occurred during training data creation: {e}")


def main():
    convert_data()
    mapped_df_path = "final_mapped_resp_rate.csv"
    pleth_output_path = "train_data.csv"
    target_output_path = "train_target.csv"

    create_training_data(mapped_df_path, pleth_output_path, target_output_path)



if __name__ == "__main__":
    main()
