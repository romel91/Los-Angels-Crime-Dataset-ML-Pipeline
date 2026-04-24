import pandas as pd
from sklearn.preprocessing import LabelEncoder


def add_time_features(df):
    df['Year']      = df['Date Occurred'].dt.year
    df['Month']     = df['Date Occurred'].dt.month
    df['DayOfWeek'] = df['Date Occurred'].dt.dayofweek
    df['Quarter']   = df['Date Occurred'].dt.quarter
    df['DayOfMonth']= df['Date Occurred'].dt.day
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    def time_bin(hour):
        if 0 <= hour < 6:     return 'Night'
        elif 6 <= hour < 12:  return 'Morning'
        elif 12 <= hour < 18: return 'Afternoon'
        else:                 return 'Evening'

    df['TimeBin'] = df['Hour Occurred'].apply(time_bin)
    df['Report_Delay_Days'] = (df['Date Reported'] - df['Date Occurred']).dt.days
    df['Report_Delay_Days'] = df['Report_Delay_Days'].clip(0, 365)
    return df


def add_frequency_encoding(df):
    df['Area_Crime_Frequency']    = df['Area ID'].map(df['Area ID'].value_counts())
    df['Premise_Crime_Frequency'] = df['Premise Code'].map(df['Premise Code'].value_counts())
    df['MO_Code_Count'] = df['MO Codes'].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    return df


def add_crime_category(df):
    def categorize(desc):
        desc = str(desc).upper()
        if any(w in desc for w in ['THEFT', 'STEAL', 'BURGLARY', 'ROBBERY', 'SHOPLIFTING']):
            return 'Property Crime'
        elif any(w in desc for w in ['ASSAULT', 'BATTERY', 'SHOOTING', 'HOMICIDE', 'MURDER']):
            return 'Violent Crime'
        elif any(w in desc for w in ['RAPE', 'SEX', 'LEWD', 'INDECENT']):
            return 'Sex Crime'
        elif any(w in desc for w in ['VEHICLE', 'AUTO', 'BIKE']):
            return 'Vehicle Crime'
        elif any(w in desc for w in ['VANDAL', 'GRAFFITI', 'ARSON']):
            return 'Vandalism'
        elif any(w in desc for w in ['DRUG', 'NARCOTIC']):
            return 'Drug Crime'
        elif any(w in desc for w in ['FRAUD', 'FORGERY', 'COUNTERFEIT', 'IDENTITY']):
            return 'Financial Crime'
        else:
            return 'Other'

    df['Crime_Category'] = df['Crime Code Description'].apply(categorize)
    return df


def encode_categoricals(df):
    cat_cols = ['Victim Sex', 'Victim Descent', 'Area Name',
                'Premise Description', 'Weapon Description',
                'TimeBin', 'Status Description']
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col + '_Enc'] = le.fit_transform(df[col].astype(str))
    return df


def run_all(df):
    df = add_time_features(df)
    df = add_frequency_encoding(df)
    df = add_crime_category(df)
    df = encode_categoricals(df)
    return df
