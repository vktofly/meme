import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load and preprocess data
def load_data(data_path="data/predictor_data.csv"):
    df = pd.read_csv(data_path)
    # Features: resource_pool, prev_action_0, prev_action_1
    X = df[["resource_pool", "prev_action_0", "prev_action_1"]].values
    # Target: action_0 (predict agent 0's action)
    y = df["action_0"].values
    return X, y

# Build model
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax")  # 6 actions (0–5)
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Main
if __name__ == "__main__":
    # Load data
    X, y = load_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build and train model
    model = build_model(input_dim=X_train.shape[1])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/predictor_model")
    np.save("models/scaler_mean.npy", scaler.mean_)
    np.save("models/scaler_scale.npy", scaler.scale_)
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv("logs/predictor_training_history.csv", index=False)