import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap
from data import load_and_preprocess_data

def evaluate_model():
    X_train, X_test, y_train, y_test, is_clf = load_and_preprocess_data()
    
    try:
        model = joblib.load('models/rf_pipeline.pkl')
    except FileNotFoundError:
        print("Model not found. Please run train_sklearn.py first.")
        return

    predictions = model.predict(X_test)
    
    probs = None
    if is_clf:
        proba_matrix = model.predict_proba(X_test)
        if proba_matrix.shape[1] > 1:
            probs = proba_matrix[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, zero_division=0))
    
    if probs is not None and len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, probs)
        print(f"ROC AUC Score: {auc:.4f}")
    
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    print("Generating SHAP explanations...")
    rf_model = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
        
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_transformed)
    
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        if len(shap_values.shape) == 3:
            shap_vals_to_plot = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
        else:
            shap_vals_to_plot = shap_values
    
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = None
        
    shap.summary_plot(shap_vals_to_plot, X_test_transformed, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
