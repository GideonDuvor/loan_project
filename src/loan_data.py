import pandas as pd

def load_data(path):
    try:
        df = pd.read_csv(path)
        print("Data loaded successfully")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")