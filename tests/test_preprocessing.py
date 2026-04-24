import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.preprocessing import clean_victim_age, clean_victim_sex, clean_victim_descent

# test_clean_victim_age
def test_clean_victim_age_removes_livestk():
    df = pd.DataFrame({'Victim Age': ['25', 'LIVESTK', '30', None, '150']})
    result = clean_victim_age(df)
    assert result['Victim Age'].isnull().sum() == 0
    assert result['Victim Age'].max() <= 100

# test_clean_victim_sex
def test_clean_victim_sex_keeps_only_mf():
    df = pd.DataFrame({'Victim Sex': ['M', 'F', 'X', None, 'H', '-']})
    result = clean_victim_sex(df)
    assert set(result['Victim Sex'].unique()) == {'M', 'F', 'Unknown'}

# test_clean_victim_descent 
def test_clean_victim_descent_maps_codes():
    df = pd.DataFrame({'Victim Descent': ['H', 'W', 'B', '-', None]})
    result = clean_victim_descent(df)
    assert 'Hispanic' in result['Victim Descent'].values
    assert 'White' in result['Victim Descent'].values
    assert 'Unknown' in result['Victim Descent'].values
