import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def print_report(y_test, y_pred, model_name):
    print(f'=== {model_name} ===')
    print(classification_report(y_test, y_pred))


def plot_confusion_matrix(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        xticks_rotation=45,
        cmap='Blues',
        ax=ax
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.tight_layout()
    plt.show()
