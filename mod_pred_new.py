import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
import joblib

# Function to load models and scaler
def load_models():
    # Load SVM model
    svm_model = joblib.load('svm2.pkl')
    
    # Load and compile ANN model
    ann_model = load_model('ann_model1.keras')
    ann_model = compile_ann_model(ann_model)
    
    # Load scaler
    scaler = joblib.load('scaler.pkl')
    return svm_model, ann_model, scaler

# Function to recompile the ANN model
def compile_ann_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to prepare data
def prepare_data(filename):
    data = pd.read_csv(filename)
    features = data.drop(['gene_id', 'Gene_type'], axis=1)
    labels = data['Gene_type'].astype(int)
    return features, labels, data['gene_id']

# Function to make predictions
def make_predictions(features, svm_model, ann_model, scaler):
    features_scaled = scaler.transform(features)
    svm_probs = svm_model.predict_proba(features_scaled)[:, 1]
    ann_probs = ann_model.predict(features_scaled).flatten()
    svm_preds = (svm_probs > 0.5).astype(int)
    ann_preds = (ann_probs > 0.5).astype(int)
    return svm_probs, ann_probs, svm_preds, ann_preds

# Function to evaluate predictions
def evaluate_predictions(labels, svm_probs, ann_probs):
    svm_auc = roc_auc_score(labels, svm_probs)
    ann_auc = roc_auc_score(labels, ann_probs)
    svm_accuracy = accuracy_score(labels, np.round(svm_probs))
    ann_accuracy = accuracy_score(labels, np.round(ann_probs))
    print(f"SVM - Accuracy: {svm_accuracy:.2f}, ROC AUC: {svm_auc:.2f}")
    print(f"ANN - Accuracy: {ann_accuracy:.2f}, ROC AUC: {ann_auc:.2f}")

# Function to plot ROC and Precision-Recall curves
def plot_evaluation_curves(labels, probs, model_name):
    # Setup
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)  # ROC Curve subplot
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} ROC (area = {roc_auc_score(labels, probs):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1, 2, 2)  # Precision-Recall Curve subplot
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.plot(recall, precision, color='green', lw=2, label=f'{model_name} Precision-Recall (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Function to save predictions to CSV
def save_predictions(gene_ids, labels, svm_probs, ann_probs, svm_preds, ann_preds, filename):
    results_df = pd.DataFrame({
        'Gene_ID': gene_ids,
        'Actual_Label': labels,
        'SVM_Probability': svm_probs,
        'ANN_Probability': ann_probs,
        'SVM_Prediction': svm_preds,
        'ANN_Prediction': ann_preds
    })
    results_df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

# Main function to run the model predictions
def main():
    svm_model, ann_model, scaler = load_models()
    features, labels, gene_ids = prepare_data('augmented_data_multiple_rounds.csv')
    svm_probs, ann_probs, svm_preds, ann_preds = make_predictions(features, svm_model, ann_model, scaler)
    evaluate_predictions(labels, svm_probs, ann_probs)
    plot_evaluation_curves(labels, svm_probs, 'SVM')
    plot_evaluation_curves(labels, ann_probs, 'ANN')
    save_predictions(gene_ids, labels, svm_probs, ann_probs, svm_preds, ann_preds, 'prediction_results.csv')

# Run the main function
if __name__ == "__main__":
    main()
