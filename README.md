# Los Angeles Crime Data Analysis (2010–2017)

## Project Overview
End-to-end ML pipeline on the Los Angeles crime dataset containing 1.58 million crime records from 2010 to 2017. The goal is to predict the **crime category** based on victim information, location, time, and other features.

---

## Dataset
- **File:** `Crime_Data_2010_2017.csv`
- **Rows:** 1,584,316
- **Columns:** 26
- **Source:** LAPD (Los Angeles Police Department)

---

## Project Structure
```
Los_angels crime Data/
├── Crime_Data_2010_2017.csv   # Raw dataset
├── crime.ipynb                # Main notebook
└── README.md                  # Project documentation
```

---

## Pipeline Steps

### 1. Data Loading
- Loaded CSV with `low_memory=False`
- Checked shape, dtypes, null counts

### 2. Data Cleaning
- **Date columns** → converted to `datetime` format
- **Time Occurred** → extracted `Hour Occurred` (0–23)
- **Victim Age** → removed non-numeric values (`LIVESTK`), clipped to 1–100, filled nulls with median
- **Victim Sex** → kept only `M`/`F`, replaced rest with `Unknown`
- **Victim Descent** → mapped single-letter codes to full names (H → Hispanic, W → White, etc.)
- **Location** → extracted `Latitude` and `Longitude` from string format `(lat, lon)`
- **Weapon columns** → filled nulls with `0` / `No Weapon`, created binary `Armed` feature
- **Dropped columns** → `Crime Code 1/2/3/4`, `Cross Street` (excessive nulls)
- **Duplicates** → removed based on unique `DR Number`

### 3. Exploratory Data Analysis (EDA)
- Top 20 crime types → Battery - Simple Assault is most frequent
- Temporal patterns → crimes peak in evening hours, Fridays have most crimes
- Victim demographics → age 20–40 most affected, Male/Female nearly equal
- Geographic scatter plot → Downtown LA has highest crime density

### 4. Feature Engineering
| Feature | Source | Description |
|---|---|---|
| `Hour Occurred` | Time Occurred | Hour of crime (0–23) |
| `Year`, `Month`, `DayOfWeek` | Date Occurred | Time-based features |
| `Quarter`, `DayOfMonth` | Date Occurred | Additional time features |
| `IsWeekend` | DayOfWeek | 1 if Saturday/Sunday |
| `TimeBin` | Hour Occurred | Night/Morning/Afternoon/Evening |
| `Report_Delay_Days` | Date Reported - Date Occurred | Days until reported |
| `Area_Crime_Frequency` | Area ID | Frequency encoding |
| `Premise_Crime_Frequency` | Premise Code | Frequency encoding |
| `MO_Code_Count` | MO Codes | Number of MO codes used |
| `Armed` | Weapon Used Code | Binary weapon flag |
| `Latitude`, `Longitude` | Location | Extracted coordinates |
| `Crime_Category` | Crime Code Description | **Target variable** (8 classes) |

### 5. Target Variable: Crime_Category
200+ crime descriptions grouped into 8 categories:
- Property Crime, Violent Crime, Vehicle Crime, Vandalism
- Sex Crime, Financial Crime, Drug Crime, Other

### 6. Model Training
Used 20% sample (~316k rows) due to hardware constraints.

| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | 40% | Baseline |
| Random Forest | 64% | `n_estimators=100`, `max_depth=15` |
| XGBoost + SMOTE | 69% | Best performance |

### 7. Class Imbalance Handling
Used **SMOTE** (Synthetic Minority Oversampling Technique) to balance all 8 classes to equal size before XGBoost training.

### 8. Top Features (Random Forest Importance)
1. MO_Code_Count
2. Armed
3. Premise Code
4. Victim Age
5. Victim Sex

---

## Libraries Used
```python
pandas, numpy, matplotlib, seaborn
scikit-learn, xgboost, imbalanced-learn
```

---

## How to Run
1. Open `crime.ipynb` in Jupyter or VS Code
2. Run all cells top to bottom
3. Ensure `Crime_Data_2010_2017.csv` is in the same directory