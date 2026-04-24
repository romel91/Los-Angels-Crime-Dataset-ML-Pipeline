from src.data import preprocessing
from src.features import engineering
from src.models.train import prepare_data, train_random_forest, train_xgboost_with_smote
from src.models.evaluate import print_report, plot_confusion_matrix

DATA_PATH = 'data/Crime_Data_2010_2017.csv'

# Main execution flow
def main():
    print('Step 1: Loading and cleaning data...')
    df = preprocessing.run_all(DATA_PATH, sample_frac=0.2)

    print('Step 2: Feature engineering...')
    df = engineering.run_all(df)

    print('Step 3: Preparing train/test split...')
    X_train, X_test, y_train, y_test = prepare_data(df)

    print('Step 4: Training Random Forest...')
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print_report(y_test, y_pred_rf, 'Random Forest')

    print('Step 5: Training XGBoost + SMOTE...')
    xgb, le = train_xgboost_with_smote(X_train, y_train)
    y_pred_xgb = le.inverse_transform(xgb.predict(X_test))
    print_report(y_test, y_pred_xgb, 'XGBoost + SMOTE')

    plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest')


if __name__ == '__main__':
    main()
