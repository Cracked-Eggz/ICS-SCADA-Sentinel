import pandas as pd
import glob
import os

def load_swat_data(data_path):
    """
    Loads and concatenates all SWaT Excel/CSV files from a folder.
    """
    print(f"Looking for files in: {data_path}")
    
    # 1. Find all Excel files
    all_files = glob.glob(os.path.join(data_path, "*.xlsx"))
    
    if not all_files:
        print("No Excel files found!")
        return None
        
    print(f"found {len(all_files)} files. Loading... (this might take a minute)")

    # 2. Loop and Load
    df_list = []
    for filename in sorted(all_files):
        print(f"  -> Reading {os.path.basename(filename)}...")
        # READ EXCEL (Engine 'openpyxl' is needed for xlsx)
        # Note: If you convert these to CSV first, it will be 10x faster!
        try:
            df_temp = pd.read_excel(filename, engine='openpyxl') 
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 3. Combine into one giant DataFrame
    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        
        # 4. Standardize Column Names
        full_df.columns = full_df.columns.str.strip()
        
        # 5. Parse Timestamp
        # Adjust 't_stamp' if your header name is different
        if 't_stamp' in full_df.columns:
            full_df['timestamp'] = pd.to_datetime(full_df['t_stamp'])
            full_df.set_index('timestamp', inplace=True)
            full_df.drop(columns=['t_stamp'], inplace=True)
            
        print(f"Loaded SWaT Data! Shape: {full_df.shape}")
        return full_df
    else:
        print("No data loaded.")
        return None

# Quick test block
if __name__ == "__main__":
    sample_path = "../../data/SWaT.A4 & A5_Jul 2019"
    df = load_swat_data(sample_path)
    if df is not None:
        print(df.head())
