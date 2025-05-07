import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model and scaler
def load_model_and_scaler(model_path="models/predictor_model", scaler_mean_path="models/scaler_mean.npy", scaler_scale_path="models/scaler_scale.npy"):
    model = tf.keras.models.load_model(model_path)
    scaler_mean = np.load(scaler_mean_path)
    scaler_scale = np.load(scaler_scale_path)
    return model, scaler_mean, scaler_scale

# Load and preprocess data
def load_data(data_path="data/predictor_data.csv"):
    df = pd.read_csv(data_path)
    X = df[["resource_pool", "prev_action_0", "prev_action_1"]].values
    y = df["action_0"].values
    return X, y

# Main
if __name__ == "__main__":
    # Load model and data
    model, scaler_mean, scaler_scale = load_model_and_scaler()
    X, y_true = load_data()
    
    # Scale data
    X_scaled = (X - scaler_mean) / scaler_scale
    
    # Generate predictions
    y_pred_probs = model.predict(X_scaled, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Compute metrics
    accuracy = np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average="weighted")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(6), yticklabels=range(6))
    plt.xlabel("Predicted Action")
    plt.ylabel("True Action")
    plt.title("Confusion Matrix for Action Prediction")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/predictor_confusion_matrix.png")
    plt.close()
    
    # Save metrics
    with open("logs/predictor_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"F1-Score: {f1:.2f}\n")