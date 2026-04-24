import numpy as np
import pandas as pd

# function to load data
def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)

# function to clean dates
def clean_dates(df):
    df['Date Reported'] = pd.to_datetime(df['Date Reported'], format='%m/%d/%Y', errors='coerce')
    df['Date Occurred'] = pd.to_datetime(df['Date Occurred'], format='%m/%d/%Y', errors='coerce')
    df['Hour Occurred'] = df['Time Occurred'].astype(str).str.zfill(4).str[:2].astype(int)
    return df

# function to clean victim age
def clean_victim_age(df):
    df['Victim Age'] = pd.to_numeric(df['Victim Age'], errors='coerce')
    df['Victim Age'] = df['Victim Age'].where(df['Victim Age'].between(1, 100), np.nan)
    df['Victim Age'] = df['Victim Age'].fillna(df['Victim Age'].median())
    return df

# function to clean victim sex
def clean_victim_sex(df):
    df['Victim Sex'] = df['Victim Sex'].where(df['Victim Sex'].isin(['M', 'F']), 'Unknown')
    return df

# function to clean victim descent
def clean_victim_descent(df):
    descent_map = {
        'A': 'Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Cambodian',
        'F': 'Filipino', 'G': 'Guamanian', 'H': 'Hispanic', 'I': 'American Indian',
        'J': 'Japanese', 'K': 'Korean', 'L': 'Laotian', 'O': 'Other',
        'P': 'Pacific Islander', 'S': 'Samoan', 'U': 'Hawaiian',
        'V': 'Vietnamese', 'W': 'White', 'X': 'Unknown', 'Z': 'Asian Indian'
    }
    df['Victim Descent'] = df['Victim Descent'].map(descent_map).fillna('Unknown')
    return df

# function to extract location
def extract_location(df):
    df['Latitude']  = df['Location'].astype(str).str.extract(r'\(([^,]+),').astype(float)
    df['Longitude'] = df['Location'].astype(str).str.extract(r',\s*([^)]+)\)').astype(float)
    df = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)]
    df.drop(columns=['Location'], inplace=True)
    return df

# function to clean weapons
def clean_weapons(df):
    df['Weapon Used Code'] = df['Weapon Used Code'].fillna(0)
    df['Weapon Description'] = df['Weapon Description'].fillna('No Weapon')
    df['Armed'] = (df['Weapon Used Code'] != 0).astype(int)
    return df

# function to drop unnecessary columns
def drop_unnecessary_columns(df):
    cols = ['Crime Code 1', 'Crime Code 2', 'Crime Code 3', 'Crime Code 4', 'Cross Street']
    existing = [c for c in cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    df['MO Codes'] = df['MO Codes'].fillna('')
    return df

# function to remove duplicates
def remove_duplicates(df):
    df.drop_duplicates(subset='DR Number', keep='first', inplace=True)
    return df

# function to fill remaining nulls
def fill_remaining_nulls(df):
    for col in ['Premise Description', 'Crime Code Description', 'Premise Code', 'Status Code']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    df = df.dropna(subset=['Latitude', 'Longitude'])
    return df

# function to run all preprocessing
def run_all(filepath, sample_frac=1.0):
    df = load_data(filepath)
    df = clean_dates(df)
    df = clean_victim_age(df)
    df = clean_victim_sex(df)
    df = clean_victim_descent(df)
    df = extract_location(df)
    df = clean_weapons(df)
    df = drop_unnecessary_columns(df)
    df = remove_duplicates(df)
    df = fill_remaining_nulls(df)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df
