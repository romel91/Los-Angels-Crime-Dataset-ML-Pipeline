import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# function to plot top n crime types
def plot_top_crimes(df, n=20):
    top = df['Crime Code Description'].value_counts().head(n)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette='viridis', legend=False)
    plt.title(f'Top {n} Crime Types in LA (2010-2017)')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.show()

# function to plot temporal patterns
def plot_temporal_patterns(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    df['Hour Occurred'].value_counts().sort_index().plot(ax=axes[0, 0], title='Crimes by Hour')
    df['DayOfWeek'].value_counts().sort_index().plot(ax=axes[0, 1], title='Crimes by Day of Week')
    df['Month'].value_counts().sort_index().plot(ax=axes[1, 0], title='Crimes by Month')
    df['Year'].value_counts().sort_index().plot(ax=axes[1, 1], title='Crimes by Year')
    plt.tight_layout()
    plt.show()

# function to plot victim demographics
def plot_victim_demographics(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(df['Victim Age'], bins=30, ax=axes[0])
    axes[0].set_title('Victim Age Distribution')
    df['Victim Sex'].value_counts().plot.pie(ax=axes[1], autopct='%1.1f%%', title='Victim Sex')
    df['Victim Descent'].value_counts().head(8).plot.bar(ax=axes[2])
    axes[2].set_title('Top 8 Victim Descent')
    plt.tight_layout()
    plt.show()

# function to plot crime locations
def plot_crime_locations(df, sample_n=50000):
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    plt.figure(figsize=(10, 12))
    plt.scatter(sample['Longitude'], sample['Latitude'], alpha=0.05, s=1, c='red')
    plt.title('Crime Locations in LA')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

# function to plot feature importance
def plot_feature_importance(model, feature_cols, top_n=10):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances.sort_values().tail(top_n).plot.barh(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()
